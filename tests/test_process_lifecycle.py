import asyncio
import os
import sys
from pathlib import Path

import pytest

import vllm_optimizer.workers.process as process_module
from vllm_optimizer.workers.process import ManagedProcess, ProcessRunner, ProcessSpec


def test_process_spec_validates_and_snapshots_environment(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        ProcessSpec(())
    with pytest.raises(TypeError):
        ProcessSpec((sys.executable,), {"BAD": 1})  # type: ignore[dict-item]
    environment = {"SYNTHETIC_VALUE": "before"}
    spec = ProcessSpec((sys.executable, "-c", "pass"), environment, tmp_path)
    environment["SYNTHETIC_VALUE"] = "after"
    assert spec.env["SYNTHETIC_VALUE"] == "before"
    assert spec.cwd == tmp_path


def test_process_output_environment_and_exit_code(tmp_path: Path) -> None:
    async def run() -> int:
        script = "import os,sys; print(os.environ['SYNTHETIC_VALUE']); print('err',file=sys.stderr); sys.exit(3)"
        process = await ProcessRunner().start(
            ProcessSpec((sys.executable, "-c", script), {"SYNTHETIC_VALUE": "visible"}, tmp_path),
            tmp_path / "process.log",
        )
        return await process.wait()

    assert asyncio.run(run()) == 3
    output = (tmp_path / "process.log").read_text(encoding="utf-8")
    assert "visible" in output and "err" in output


def test_process_capture_stream_and_timeout_termination(tmp_path: Path) -> None:
    async def run() -> int:
        process = await ProcessRunner(capture=True).start(
            ProcessSpec((sys.executable, "-c", "import time; print('start',flush=True); time.sleep(30)")),
            tmp_path / "sleep.log",
        )
        await asyncio.sleep(0.1)
        process.write_log("marker\n")
        return await process.stop(grace_period=0.05)

    assert isinstance(asyncio.run(run()), int)
    assert "marker" in (tmp_path / "sleep.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("force", [False, True])
def test_posix_stop_checks_group_after_parent_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, force: bool
) -> None:
    class ExitedProcess:
        pid = 123
        returncode = 0

        async def wait(self) -> int:
            return 0

    alive = True
    signals: list[int] = []

    def kill_group(pid: int, sent: int) -> None:
        nonlocal alive
        assert pid == 123
        if sent == 0:
            if alive:
                return
            raise ProcessLookupError
        signals.append(sent)
        if not force or sent == process_module._SIGKILL:
            alive = False

    monkeypatch.setattr(process_module, "_POSIX", True)
    monkeypatch.setattr(process_module, "_KILLPG", kill_group)
    monkeypatch.setattr(process_module, "_SIGKILL", 9)
    with (tmp_path / "exited.log").open("w", encoding="utf-8") as log:
        managed = ManagedProcess(ExitedProcess(), log)  # type: ignore[arg-type]
        assert asyncio.run(managed.stop(0)) == 0

    assert signals[0] == process_module.signal.SIGTERM
    assert len(signals) == (2 if force else 1)


@pytest.mark.skipif(os.name != "posix", reason="POSIX process-group behavior")
def test_stop_signals_surviving_group_after_parent_exits(tmp_path: Path) -> None:
    async def run() -> int:
        ready = tmp_path / "child.ready"
        stopped = tmp_path / "child.stopped"
        child = (
            "import signal,sys,time; from pathlib import Path; "
            f"ready=Path({str(ready)!r}); stopped=Path({str(stopped)!r}); "
            "signal.signal(signal.SIGTERM, lambda *_: (stopped.write_text('stopped'), sys.exit(0))); "
            "ready.write_text('ready'); time.sleep(30)"
        )
        parent = (
            "import subprocess,time; from pathlib import Path; "
            f"subprocess.Popen([{sys.executable!r}, '-c', {child!r}]); ready=Path({str(ready)!r}); "
            "\nwhile not ready.exists(): time.sleep(0.01)"
        )
        process = await ProcessRunner().start(ProcessSpec((sys.executable, "-c", parent)), tmp_path / "parent.log")
        while process.returncode is None:
            await asyncio.sleep(0.01)
        return await process.stop(0.5)

    assert asyncio.run(run()) == 0
    assert (tmp_path / "child.stopped").read_text(encoding="utf-8") == "stopped"
