"""Committed results and recoverable checkpoints for a single GPU writer."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import uuid

from lia_gpu_four_group_search_v2 import file_hash


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex[:12])
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def status(path, value):
    for attempt in range(3):
        try:
            atomic_json(path, value)
            return True
        except OSError:
            time.sleep(0.01 * (attempt + 1))
    try:
        atomic_json(Path(path).with_name("status-fallback-" + uuid.uuid4().hex + ".json"), value)
    except OSError:
        pass
    return False


class Store:
    def __init__(self, root):
        self.root = Path(root)

    def directory(self, kind, key):
        if kind not in ("spin", "forward", "checkpoint") or len(key) != 64 or any(
                char not in "0123456789ABCDEF" for char in key):
            raise ValueError("invalid cache identity")
        return self.root / kind / key[:16]

    def read(self, kind, key):
        directory = self.directory(kind, key)
        path = directory / "result.json"
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value["key"] != key:
            raise ValueError("cache prefix collision")
        for artifact in value.get("artifacts", []):
            if Path(artifact["name"]).name != artifact["name"]:
                raise ValueError("invalid artifact name")
            if file_hash(directory / artifact["name"]) != artifact["sha256"]:
                raise ValueError("cache artifact hash mismatch")
        return value

    def arrays(self, kind, key):
        import numpy as np
        value = self.read(kind, key)
        if value is None or not value.get("artifacts"):
            raise ValueError("committed state missing")
        with np.load(self.directory(kind, key) / value["artifacts"][0]["name"], allow_pickle=False) as data:
            return {name: data[name].copy() for name in data.files}

    def commit(self, kind, key, value, arrays=None):
        import numpy as np
        if self.read(kind, key) is not None:
            raise FileExistsError("completed fingerprint cannot be recomputed")
        directory = self.directory(kind, key)
        directory.mkdir(parents=True, exist_ok=True)
        artifacts = []
        if arrays is not None:
            blob = directory / ("state-" + uuid.uuid4().hex[:12] + ".npz")
            with blob.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
                stream.flush()
                os.fsync(stream.fileno())
            artifacts.append({"name": blob.name, "sha256": file_hash(blob)})
        result = {**value, "key": key, "artifacts": artifacts}
        atomic_json(directory / "result.json", result)
        return result

    def checkpoint(self, kind, key, year, value, arrays):
        target = Store(self.root / "resume" / kind / key[:16] / f"y{year:04d}")
        return target.commit("checkpoint", key, {**value, "year": year}, arrays)

    def latest(self, kind, key):
        parent = self.root / "resume" / kind / key[:16]
        for directory in sorted(parent.glob("y[0-9][0-9][0-9][0-9]"), reverse=True):
            target = Store(directory)
            result = target.read("checkpoint", key)
            if result is not None:
                return result, target.arrays("checkpoint", key)
        return None


class WriterLock:
    """The OS releases this lock if the controller dies; no PID guessing."""
    def __init__(self, root):
        self.path = Path(root) / "writer.lock"

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open("a+b")
        self.stream.seek(0, 2)
        if self.stream.tell() == 0:
            self.stream.write(b"0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.stream.close()
            raise RuntimeError("another writer owns this output root")
        return self

    def __exit__(self, *args):
        self.stream.close()
