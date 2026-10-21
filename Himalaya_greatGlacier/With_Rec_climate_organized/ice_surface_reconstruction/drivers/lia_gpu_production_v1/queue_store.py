"""No-model, single-writer queue evidence for an independently authorized run.

This module grants no launch/retry authority and imports no numerical library.
It uses the OS-released writer-lock and fsync/unique-temp patterns of
``lia_gpu_matrix_io_v2`` without importing that module's search dependencies.
Terminal manifests are published with a no-overwrite hard link on local NTFS
(or another local filesystem supporting atomic hard links). Do not use this
store on a filesystem that does not provide these semantics.

The caller owns artifact production and scientific validation. A complete
record means a technically committed stage, NOT a scientifically accepted
glacier. A technical failure is terminal for this transaction; it is never an
automatic retry. Corrupt records are isolated and never silently recomputed.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
import uuid


class QueueStoreError(RuntimeError):
    """Invalid queue evidence or use of the queue outside its writer lease."""


class FingerprintMismatch(QueueStoreError):
    pass


class LeaseConflict(QueueStoreError):
    pass


class CorruptRecordError(QueueStoreError):
    pass


class TerminalExistsError(QueueStoreError):
    pass


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _fingerprint(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-fA-F]{64}", value) is None:
        raise ValueError("execution_fingerprint must be a SHA-256 hex digest")
    return value.upper()


def _identity(key, stage):
    if not isinstance(key, str) or not key.strip() or len(key) > 512 or "\0" in key:
        raise ValueError("member key must be a nonempty string of at most 512 characters")
    if not isinstance(stage, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", stage) is None:
        raise ValueError("invalid stage name")
    return {"key": key, "stage": stage}


class QueueStore:
    """A frozen membership ledger plus one immutable terminal per member/stage.

    Construction is read-only. Mutations require ``with store.acquire_lease():``.
    ``register_members`` freezes an ordered iterable of ``(key, stage)`` pairs.
    ``recovery_inventory`` independently verifies terminal records/artifacts;
    its ``pending`` list alone is dispatchable. ``blocked`` needs adjudication,
    and ``technical_failures`` must NOT be added back to that pending list.
    The lease guards one controller, not independent worker processes; workers
    should return artifacts for that controller to commit.
    """

    def __init__(self, root, execution_fingerprint):
        self.root = Path(root).resolve()
        if self.root in (Path(self.root.anchor), Path.home().resolve(), Path.cwd().resolve()):
            raise ValueError("queue root must be a dedicated output subdirectory")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("queue root is not a directory")
        self.execution_fingerprint = _fingerprint(execution_fingerprint)
        self._lease_stream = None
        self._lease_pid = None

    def _path(self, relative):
        relative = Path(relative)
        if relative.is_absolute() or relative.drive or ".." in relative.parts:
            raise ValueError("queue path must be relative and cannot traverse parents")
        path = self.root / relative
        if not path.resolve().is_relative_to(self.root):
            raise ValueError("queue path escapes root through a link")
        return path

    def _assert_writer(self):
        if (self._lease_stream is None or self._lease_stream.closed
                or self._lease_pid != os.getpid()):
            raise LeaseConflict("mutation requires this process's active writer lease")

    @contextmanager
    def acquire_lease(self):
        """Acquire a nonblocking OS lock; a dead process releases it automatically."""
        if self._lease_stream is not None:
            raise LeaseConflict("this store already holds a writer lease")
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self._path("writer.lock")
        stream = lock_path.open("a+b")
        try:
            stream.seek(0, 2)
            if stream.tell() == 0:
                stream.write(b"0")
                stream.flush()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise LeaseConflict("another controller owns this output root") from exc
            self._lease_stream, self._lease_pid = stream, os.getpid()
            identity = {"schema": "lia-gpu-production-queue-v1",
                        "execution_fingerprint": self.execution_fingerprint}
            path = self._path("execution.json")
            if path.exists():
                existing = self._read_document(path)
                if existing != identity:
                    raise FingerprintMismatch("output root belongs to another execution")
            else:
                # Never adopt an already populated, unbound output directory.
                if any(p.name != "writer.lock" for p in self.root.iterdir()):
                    raise QueueStoreError("refusing to adopt a nonempty unbound queue root")
                self._publish_document(path, identity)
            yield self
        finally:
            stream.close()
            if self._lease_stream is stream:
                self._lease_stream = self._lease_pid = None

    def _publish_document(self, path, payload):
        """Flush before atomic publication; never replace an existing record."""
        self._assert_writer()
        payload = json.loads(_json_bytes(payload))  # detach mutable caller data
        document = {"payload": payload,
                    "sha256": hashlib.sha256(_json_bytes(payload)).hexdigest().upper()}
        self._path(path.relative_to(self.root))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
        try:
            with temporary.open("xb") as stream:
                stream.write(_json_bytes(document))
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)  # atomic and fails if destination exists
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass  # an orphan temp file is not a committed record

    def _read_document(self, path):
        try:
            self._path(path.relative_to(self.root))
            document = json.loads(path.read_text(encoding="utf-8"))
            payload = document["payload"]
            expected = hashlib.sha256(_json_bytes(payload)).hexdigest().upper()
            if document["sha256"] != expected:
                raise ValueError("document digest mismatch")
            if payload["execution_fingerprint"] != self.execution_fingerprint:
                raise FingerprintMismatch("record belongs to another execution")
            return payload
        except FingerprintMismatch:
            raise
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CorruptRecordError(f"invalid record {path.name}: {exc}") from exc

    def register_members(self, members):
        """Freeze membership/order once; identical registration is idempotent."""
        self._assert_writer()
        rows = [_identity(key, stage) for key, stage in members]
        identities = [(r["key"], r["stage"]) for r in rows]
        if not rows or len(set(identities)) != len(identities):
            raise ValueError("membership must be nonempty and unique per key/stage")
        payload = {"schema": "lia-gpu-production-members-v1",
                   "execution_fingerprint": self.execution_fingerprint, "members": rows}
        path = self._path("members.json")
        if path.exists():
            if self._read_document(path) != payload:
                raise QueueStoreError("frozen queue membership cannot be changed")
        else:
            self._publish_document(path, payload)
        return rows

    def _members(self):
        self._read_document(self._path("execution.json"))
        rows = self._read_document(self._path("members.json"))["members"]
        try:
            checked = [_identity(r["key"], r["stage"]) for r in rows]
            identities = [(r["key"], r["stage"]) for r in checked]
            if not rows or len(set(identities)) != len(identities) or rows != checked:
                raise ValueError("invalid frozen membership")
        except (ValueError, KeyError, TypeError) as exc:
            raise CorruptRecordError("invalid frozen membership") from exc
        return checked

    def _record_path(self, key, stage):
        identity = _identity(key, stage)
        digest = hashlib.sha256(_json_bytes(identity)).hexdigest()
        return self._path(Path("members") / digest / "terminal.json")

    def artifact_ref(self, path):
        """Hash an existing in-root artifact without loading numerical arrays."""
        path = Path(path)
        if path.is_absolute():
            try:
                path = path.relative_to(self.root)
            except ValueError as exc:
                raise ValueError("artifact must be inside the queue root") from exc
        checked = self._path(path)
        if not checked.is_file():
            raise ValueError("artifact must be an existing regular file")
        return {"path": checked.relative_to(self.root).as_posix(),
                "sha256": _sha256(checked), "size_bytes": checked.stat().st_size}

    def _artifact_refs(self, artifacts):
        if isinstance(artifacts, (str, bytes, os.PathLike)):
            raise TypeError("artifacts must be an iterable of paths or artifact references")
        refs = []
        for artifact in artifacts:
            if isinstance(artifact, dict):
                actual = self.artifact_ref(artifact["path"])
                if artifact != actual:
                    raise CorruptRecordError("artifact reference does not match bytes on disk")
            else:
                actual = self.artifact_ref(artifact)
            refs.append(actual)
        if len({r["path"] for r in refs}) != len(refs):
            raise ValueError("duplicate artifact paths")
        return refs

    def _verify_terminal(self, path, key, stage):
        value = self._read_document(path)
        if (value.get("schema") != "lia-gpu-production-terminal-v1"
                or value.get("key") != key or value.get("stage") != stage
                or value.get("status") not in ("complete", "technical-failure")):
            raise CorruptRecordError("terminal identity or status mismatch")
        try:
            artifacts = self._artifact_refs(value["artifacts"])
            if value["status"] == "complete" and not artifacts:
                raise ValueError("complete stage has no artifact")
            if value["status"] == "technical-failure" and not value.get("reason"):
                raise ValueError("technical failure has no reason")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise CorruptRecordError(f"terminal artifact invalid: {exc}") from exc
        return value

    def _commit(self, key, stage, status, artifacts, metadata, reason=None):
        self._assert_writer()
        if _identity(key, stage) not in self._members():
            raise ValueError("member/stage is outside frozen membership")
        path = self._record_path(key, stage)
        if path.exists():
            self._verify_terminal(path, key, stage)
            raise TerminalExistsError("terminal member must not be recomputed or overwritten")
        refs = self._artifact_refs(artifacts)
        if status == "complete" and not refs:
            raise ValueError("complete stage requires at least one committed artifact")
        value = {"schema": "lia-gpu-production-terminal-v1",
                 "execution_fingerprint": self.execution_fingerprint,
                 "key": key, "stage": stage, "status": status,
                 "created_at": _now(), "artifacts": refs,
                 "metadata": {} if metadata is None else metadata}
        if reason is not None:
            value["reason"] = reason
        self._publish_document(path, value)
        return self._verify_terminal(path, key, stage)

    def commit_complete(self, key, stage, artifacts, metadata=None):
        """Commit a technically complete stage; refuses all prior terminal records."""
        return self._commit(key, stage, "complete", artifacts, metadata)

    def commit_technical_failure(self, key, stage, reason, artifacts=(), metadata=None):
        """Isolate one failed member; no implicit retry or scientific rejection."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a technical failure requires a nonempty reason")
        return self._commit(key, stage, "technical-failure", artifacts, metadata, reason)

    def _inspect_registered(self, key, stage):
        row = {**_identity(key, stage), "state": "pending", "record": None, "error": None}
        try:
            path = self._record_path(key, stage)
            if path.exists():
                row["record"] = self._verify_terminal(path, key, stage)
                row["state"] = row["record"]["status"]
        except (QueueStoreError, OSError, ValueError) as exc:
            row.update(state="blocked-corrupt", record=None, error=str(exc))
        return row

    def inspect_member(self, key, stage):
        if _identity(key, stage) not in self._members():
            raise ValueError("member/stage is outside frozen membership")
        return self._inspect_registered(key, stage)

    def recovery_inventory(self):
        """Recount from immutable records, never from mutable status.json."""
        result = {"pending": [], "complete": [], "technical_failures": [], "blocked": []}
        categories = {"pending": "pending", "complete": "complete",
                      "technical-failure": "technical_failures", "blocked-corrupt": "blocked"}
        for member in self._members():
            row = self._inspect_registered(**member)
            result[categories[row["state"]]].append(row)
        return result

    def pending_members(self):
        """Return dispatchable identities only; inspect inventory for isolated errors."""
        return [(r["key"], r["stage"]) for r in self.recovery_inventory()["pending"]]

    def write_status(self, payload):
        """Best-effort monitoring; I/O/serialization failure never kills computation.

        Return ``updated``, ``durable``, ``snapshot_path`` and ``errors``. If the
        entire disk is unavailable, errors remain in this return value and the
        caller's computation can continue. Lease/programming violations still
        raise rather than permitting a second mutable-status writer.
        """
        self._assert_writer()
        errors = []
        value = {"execution_fingerprint": self.execution_fingerprint,
                 "observed_at": _now(), "status": payload}
        for attempt in range(3):
            temporary = None
            try:
                path = self._path("status.json")
                temporary = self._path("status.json.tmp." + uuid.uuid4().hex)
                with temporary.open("xb") as stream:
                    stream.write(_json_bytes(value))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                return {"updated": True, "durable": True,
                        "snapshot_path": str(path), "errors": errors}
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"status attempt {attempt + 1}: {type(exc).__name__}: {exc}")
                if attempt < 2:
                    time.sleep(0.01 * (attempt + 1))
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
        try:
            fallback = self._path(Path("status-snapshots") / (uuid.uuid4().hex + ".json"))
            self._publish_document(fallback, {**value, "status_write_errors": errors})
            return {"updated": False, "durable": True,
                    "snapshot_path": str(fallback), "errors": errors}
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"status fallback: {type(exc).__name__}: {exc}")
            return {"updated": False, "durable": False, "snapshot_path": None, "errors": errors}
