from pathlib import Path

from vllm_optimizer.workers.endpoint_ownership import _listener_inodes


def test_listener_inode_parser_selects_listening_port(tmp_path: Path) -> None:
    net = tmp_path / "net"
    net.mkdir()
    (net / "tcp").write_text(
        "header\n"
        "0: 0100007F:1F40 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 12345\n"
        "1: 0100007F:1F41 00000000:0000 0A 00000000:00000000 00:00000000 00000000 1000 0 99999\n",
        encoding="ascii",
    )

    assert _listener_inodes(tmp_path, 8000) == {"12345"}
