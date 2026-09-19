"""Linux listener ownership checks for managed server processes."""

from __future__ import annotations

import os
import re
from pathlib import Path

_SOCKET = re.compile(r"socket:\[(\d+)\]")


def process_owns_listener(pid: int, port: int, proc: Path = Path("/proc")) -> bool:
    """Return whether a process session owns a listening TCP socket."""
    listeners = _listener_inodes(proc, port)
    if not listeners:
        return False
    return any(listeners.intersection(_socket_inodes(proc / str(member) / "fd")) for member in _session_pids(proc, pid))


def _listener_inodes(proc: Path, port: int) -> set[str]:
    inodes: set[str] = set()
    expected = f"{port:04X}"
    for name in ("tcp", "tcp6"):
        try:
            lines = (proc / "net" / name).read_text(encoding="ascii").splitlines()[1:]
        except (OSError, UnicodeError):
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[1].rsplit(":", 1)[-1].upper() == expected and fields[3] == "0A":
                inodes.add(fields[9])
    return inodes


def _session_pids(proc: Path, leader: int) -> tuple[int, ...]:
    members = []
    try:
        entries = tuple(proc.iterdir())
    except OSError:
        return ()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text(encoding="ascii")
            fields = stat[stat.rfind(")") + 2 :].split()
            if len(fields) > 3 and int(fields[3]) == leader:
                members.append(int(entry.name))
        except (OSError, UnicodeError, ValueError):
            continue
    return tuple(members)


def _socket_inodes(directory: Path) -> set[str]:
    inodes: set[str] = set()
    try:
        entries = tuple(directory.iterdir())
    except OSError:
        return inodes
    for entry in entries:
        try:
            match = _SOCKET.fullmatch(os.readlink(entry))
        except OSError:
            continue
        if match:
            inodes.add(match.group(1))
    return inodes
