from pathlib import Path

from vllm_optimizer.workers.endpoint_ownership import _listener_inodes


def test_listener_inode_parser_selects_address_and_port(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp").write_text(
        "header\n"
        "0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 12345\n"
        "1: 0200007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 99999\n",
        encoding="ascii",
    )

    assert _listener_inodes(tmp_path, "127.0.0.1", 8000) == {"12345"}
    assert _listener_inodes(tmp_path, "127.0.0.2", 8000) == {"99999"}


def test_listener_inode_parser_accepts_wildcard_address(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp").write_text(
        "header\n0: 00000000:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 12345\n",
        encoding="ascii",
    )

    assert _listener_inodes(tmp_path, "127.0.0.1", 8000) == {"12345"}


def test_listener_inode_parser_decodes_ipv6_address(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp6").write_text(
        "header\n"
        "0: 00000000000000000000000001000000:1F40 00000000000000000000000000000000:0000 "
        "0A 00000000:00000000 00:00000000 00000000 1000 0 12345\n",
        encoding="ascii",
    )

    assert _listener_inodes(tmp_path, "::1", 8000) == {"12345"}


def test_ipv6_wildcard_does_not_match_ipv4_health_address(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp6").write_text(
        "header\n"
        "0: 00000000000000000000000000000000:1F40 00000000000000000000000000000000:0000 "
        "0A 00000000:00000000 00:00000000 00000000 1000 0 12345\n",
        encoding="ascii",
    )

    assert _listener_inodes(tmp_path, "127.0.0.1", 8000) == set()
