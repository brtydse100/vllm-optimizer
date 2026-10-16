"""Plan gated releases and attach immutable distributions to their exact commit."""

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import tomllib
from pathlib import Path


def command(*arguments: str, cwd: Path | None = None) -> str:
    return subprocess.run(arguments, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def metadata(root: Path) -> tuple[str, Path]:
    version = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", version):
        raise ValueError("unsupported release version")
    tree = ast.parse((root / "src/vllm_optimizer/__init__.py").read_text(encoding="utf-8"))
    runtime = next(
        (
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "__version__" for target in node.targets)
        ),
        None,
    )
    if runtime != version:
        raise ValueError("package and runtime version disagree")
    notes = root / f"docs/project/releases/v{version}.md"
    if not notes.is_file():
        raise ValueError("release notes are missing")
    return version, notes


def prepare(root: Path, event: str, ref: str, before: str) -> dict[str, str]:
    version, _ = metadata(root)
    sha = command("git", "rev-parse", "HEAD", cwd=root)
    enabled = True
    if event == "push":
        if not re.fullmatch(r"[0-9a-f]{40}", before) or set(before) == {"0"}:
            raise ValueError("a previous commit is required for merge releases")
        previous = tomllib.loads(command("git", "show", f"{before}:pyproject.toml", cwd=root))
        enabled = previous["project"]["version"] != version
    elif event not in {"release", "workflow_dispatch"} or ref != f"v{version}":
        raise ValueError("manual releases require the matching immutable version tag")
    return {"enabled": str(enabled).lower(), "sha": sha, "tag": f"v{version}", "version": version}


def gh(*arguments: str, missing: bool = False) -> str | None:
    process = subprocess.run(("gh", *arguments), capture_output=True, text=True)
    if process.returncode:
        if missing and "HTTP 404" in process.stderr:
            return None
        raise RuntimeError(process.stderr.strip() or "GitHub command failed")
    return process.stdout


def remote_commit(repository: str, tag: str) -> str | None:
    document = gh("api", f"repos/{repository}/git/ref/tags/{tag}", missing=True)
    if document is None:
        return None
    value = json.loads(document)["object"]
    while value["type"] == "tag":
        value = json.loads(gh("api", f"repos/{repository}/git/tags/{value['sha']}"))["object"]
    if value["type"] != "commit":
        raise ValueError("release tag does not identify a commit")
    return str(value["sha"])


def publish(root: Path, sha: str, tag: str, repository: str) -> None:
    version, notes = metadata(root)
    if command("git", "rev-parse", "HEAD", cwd=root) != sha or tag != f"v{version}":
        raise ValueError("release commit or tag disagrees with the checked-out package")
    assets = [
        root / "evidence" / name
        for name in (
            f"dist/vllm_optimizer-{version}-py3-none-any.whl",
            f"dist/vllm_optimizer-{version}.tar.gz",
            "sbom.json",
        )
    ]
    if not all(path.is_file() for path in assets):
        raise ValueError("required release artifact is missing")
    existing = remote_commit(repository, tag)
    if existing is not None and existing != sha:
        raise ValueError("existing release tag points to another commit")
    document = gh("api", f"repos/{repository}/releases/tags/{tag}", missing=True)
    release = json.loads(document) if document is not None else None
    recorded = {asset["name"]: asset.get("digest") for asset in release["assets"]} if release else {}
    for path in assets:
        if path.name in recorded and recorded[path.name] != "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("existing release artifact has a different or unavailable digest")
    if existing is None:
        gh("api", f"repos/{repository}/git/refs", "--method", "POST", "-f", f"ref=refs/tags/{tag}", "-f", f"sha={sha}")
    if remote_commit(repository, tag) != sha:
        raise ValueError("release tag commit changed")
    if release is None:
        title = notes.read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
        gh(
            "release",
            "create",
            tag,
            "--verify-tag",
            "--title",
            title,
            "--notes-file",
            str(notes),
            *(("--prerelease",) if re.search(r"[a-z]", version) else ()),
            *(str(path) for path in assets),
        )
    else:
        for path in assets:
            if path.name not in recorded:
                gh("release", "upload", tag, str(path))


if __name__ == "__main__":
    root = Path.cwd()
    if sys.argv[1:] == ["plan"]:
        plan = prepare(
            root, os.environ["RELEASE_EVENT"], os.environ.get("RELEASE_REF", ""), os.environ.get("RELEASE_BEFORE", "")
        )
        with Path(os.environ["GITHUB_OUTPUT"]).open("a", encoding="utf-8") as stream:
            stream.write("".join(f"{key}={value}\n" for key, value in plan.items()))
        print(json.dumps(plan))
    elif sys.argv[1:] == ["publish"]:
        publish(root, os.environ["RELEASE_COMMIT"], os.environ["RELEASE_TAG"], os.environ["GITHUB_REPOSITORY"])
    else:
        raise SystemExit("usage: release.py plan|publish")
