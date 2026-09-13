"""Download and verify the exact small model used by both GPU experiments."""

import hashlib
from pathlib import Path

from huggingface_hub import snapshot_download

MODEL = Path("/var/tmp/vllm-optimizer-demo/models/opt-125m")
REVISION = "27dcfa74d334bc871f3234de431e71c6eeba5dd6"


def main() -> None:
    checksums = Path(__file__).with_name("model.sha256").read_text(encoding="utf-8").splitlines()
    expected = dict(line.split("  ", 1)[::-1] for line in checksums if line.strip())
    snapshot_download("facebook/opt-125m", revision=REVISION, local_dir=MODEL, allow_patterns=list(expected))
    for name, checksum in expected.items():
        with (MODEL / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != checksum:
            raise SystemExit(f"Model checksum mismatch: {name}")
    print(f"Verified {len(expected)} model/tokenizer files at {MODEL}")


if __name__ == "__main__":
    main()
