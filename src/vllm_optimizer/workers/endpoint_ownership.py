"""Linux listener ownership checks for managed server processes."""

from __future__ import annotations

import ipaddress
import os
import re
import socket
from pathlib import Path

_SOCKET = re.compile(r"socket:\[(\d+)\]")
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def process_owns_listener(pid: int, host: str, port: int, proc: Path = Path("/proc")) -> bool:
    """Return whether a process session owns the addressed listening socket."""
    listeners = _listener_inodes(proc, host, port)
    if not listeners:
        return False
    return any(listeners.intersection(_socket_inodes(proc / str(member) / "fd")) for member in _session_pids(proc, pid))


def _listener_inodes(proc: Path, host: str, port: int) -> set[str]:
    inodes: set[str] = set()
    expected = f"{port:04X}"
    addresses = _resolved_addresses(host)
    for name, version in (("tcp", 4), ("tcp6", 6)):
        try:
            lines = (proc / "net" / name).read_text(encoding="ascii").splitlines()[1:]
        except (OSError, UnicodeError):
            continue
        for line in lines:
            fields = line.split()
            local_address, local_port = fields[1].rsplit(":", 1) if len(fields) > 9 else ("", "")
            address = _decoded_address(local_address, version)
            if (
                address is not None
                and local_port.upper() == expected
                and fields[3] == "0A"
                and (
                    address in addresses
                    or address.is_unspecified
                    and any(item.version == version for item in addresses)
                )
            ):
                inodes.add(fields[9])
    return inodes


def _resolved_addresses(host: str) -> set[IPAddress]:
    try:
        records = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return set()
    return {ipaddress.ip_address(str(record[4][0]).split("%", 1)[0]) for record in records}


def _decoded_address(encoded: str, version: int) -> IPAddress | None:
    try:
        raw = bytes.fromhex(encoded)
        packed = raw[::-1] if version == 4 else b"".join(raw[index : index + 4][::-1] for index in range(0, 16, 4))
        return ipaddress.ip_address(packed)
    except ValueError:
        return None


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
