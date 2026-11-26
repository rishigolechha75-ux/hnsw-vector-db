"""Append-only write-ahead log with CRC32 framing and torn-write recovery.

Frame layout (little endian)::

    u32 body_length | u32 crc32(body) | body

    body = u64 lsn | u8 op | u32 header_len | header_json | vector bytes (float32)

* ``lsn`` is a monotonically increasing log sequence number. Snapshots remember
  the last LSN they include, so replay after a crash between "snapshot written" and
  "log truncated" skips records that are already in the snapshot.
* On open, frames are read until the first one that is incomplete or fails its CRC.
  That tail is the result of a crash mid-write (a "torn write"); it is truncated so
  the next append starts on a clean boundary. Everything before it is intact.
"""
from __future__ import annotations

import fcntl
import json
import os
import struct
import zlib
from dataclasses import dataclass, field
from typing import Iterator, List, Optional

import numpy as np

OP_UPSERT = 1
OP_DELETE = 2

_FRAME = struct.Struct("<II")
_BODY = struct.Struct("<QBI")


@dataclass
class Record:
    lsn: int
    op: int
    header: dict
    vector: Optional[np.ndarray] = None


@dataclass
class ReplayResult:
    records: List[Record] = field(default_factory=list)
    truncated_bytes: int = 0


def encode_record(rec: Record) -> bytes:
    header = json.dumps(rec.header, separators=(",", ":")).encode()
    vec = b"" if rec.vector is None else np.asarray(rec.vector, np.float32).tobytes()
    body = _BODY.pack(rec.lsn, rec.op, len(header)) + header + vec
    return _FRAME.pack(len(body), zlib.crc32(body)) + body


def decode_body(body: bytes) -> Record:
    lsn, op, hlen = _BODY.unpack_from(body, 0)
    off = _BODY.size
    header = json.loads(body[off:off + hlen].decode())
    rest = body[off + hlen:]
    vec = np.frombuffer(rest, dtype=np.float32).copy() if rest else None
    return Record(lsn, op, header, vec)


def iter_frames(data: bytes) -> Iterator[tuple]:
    """Yield ``(end_offset, Record)`` for each valid frame; stop at the first bad one."""
    pos = 0
    n = len(data)
    while pos + _FRAME.size <= n:
        length, crc = _FRAME.unpack_from(data, pos)
        start = pos + _FRAME.size
        end = start + length
        if end > n:
            return
        body = data[start:end]
        if zlib.crc32(body) != crc:
            return
        try:
            rec = decode_body(body)
        except (ValueError, struct.error, UnicodeDecodeError):
            return
        yield end, rec
        pos = end


def durable_sync(fd: int) -> None:
    """Force data to stable storage.

    On macOS ``fsync`` only hands data to the drive, which may keep it in a volatile
    cache; ``F_FULLFSYNC`` asks the drive to flush that cache too. Linux ``fsync``
    already does this.
    """
    if hasattr(fcntl, "F_FULLFSYNC"):
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
            return
        except OSError:
            pass
    os.fsync(fd)


class WriteAheadLog:
    """Durable append-only log. Use :meth:`replay` once at startup, then :meth:`append`.

    Args:
        path: log file path (created if missing).
        fsync: if True every :meth:`append` calls ``os.fsync`` (durable against power
            loss, much slower). If False data reaches the OS page cache on each append,
            which survives a process crash but not a machine crash.
    """

    def __init__(self, path: str, fsync: bool = False):
        self.path = path
        self.fsync = fsync
        self._f = None

    def replay(self) -> ReplayResult:
        """Read all intact records and truncate any torn tail."""
        result = ReplayResult()
        if not os.path.exists(self.path):
            open(self.path, "wb").close()
            return result
        with open(self.path, "rb") as f:
            data = f.read()
        good = 0
        for end, rec in iter_frames(data):
            result.records.append(rec)
            good = end
        if good < len(data):
            result.truncated_bytes = len(data) - good
            with open(self.path, "r+b") as f:
                f.truncate(good)
        return result

    def _file(self):
        if self._f is None:
            self._f = open(self.path, "ab")
        return self._f

    def append(self, records: List[Record]) -> None:
        """Write records as one buffered write, then flush (and fsync if enabled)."""
        if not records:
            return
        f = self._file()
        f.write(b"".join(encode_record(r) for r in records))
        f.flush()
        if self.fsync:
            durable_sync(f.fileno())

    def reset(self) -> None:
        """Truncate the log (called after a successful snapshot)."""
        self.close()
        with open(self.path, "wb") as f:
            f.flush()
            os.fsync(f.fileno())

    def size_bytes(self) -> int:
        return os.path.getsize(self.path) if os.path.exists(self.path) else 0

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None
