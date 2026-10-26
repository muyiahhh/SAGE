"""Checkpoint publishing shared by the ISG and PSA trainers."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def staged_checkpoint(save_dir: Path, prefix: str = "sage_ckpt_") -> Iterator[Path]:
    """Yield a local staging directory and publish it atomically to ``save_dir``.

    Files are written under the system temporary directory first and copied to
    the final location afterwards, which avoids partial checkpoints on network
    filesystems. A failed copy never leaves a directory that looks complete.
    """
    save_dir = Path(save_dir)
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp_root:
        tmp_dir = Path(tmp_root) / "checkpoint"
        tmp_dir.mkdir()
        yield tmp_dir

        save_dir.parent.mkdir(parents=True, exist_ok=True)
        partial_dir = save_dir.with_name(f".{save_dir.name}.partial-{os.getpid()}")
        shutil.rmtree(partial_dir, ignore_errors=True)
        try:
            shutil.copytree(tmp_dir, partial_dir)
            if save_dir.exists():
                shutil.rmtree(save_dir)
            partial_dir.replace(save_dir)
        except BaseException:
            shutil.rmtree(partial_dir, ignore_errors=True)
            raise
