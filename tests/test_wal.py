import os

import numpy as np

from vecdb.wal import OP_DELETE, OP_UPSERT, Record, WriteAheadLog, encode_record


def _recs(n):
    return [Record(i + 1, OP_UPSERT, {"id": str(i), "payload": {"k": i}},
                   np.full(3, i, np.float32)) for i in range(n)]


def test_roundtrip(tmp_path):
    p = str(tmp_path / "w.log")
    w = WriteAheadLog(p)
    assert w.replay().records == []
    w.append(_recs(5) + [Record(6, OP_DELETE, {"id": "2"})])
    w.close()
    out = WriteAheadLog(p).replay()
    assert [r.lsn for r in out.records] == [1, 2, 3, 4, 5, 6]
    assert out.records[3].header == {"id": "3", "payload": {"k": 3}}
    np.testing.assert_array_equal(out.records[3].vector, [3, 3, 3])
    assert out.records[5].vector is None and out.truncated_bytes == 0


def test_torn_tail_is_truncated_and_log_reusable(tmp_path):
    p = str(tmp_path / "w.log")
    w = WriteAheadLog(p)
    w.append(_recs(3))
    w.close()
    good = os.path.getsize(p)
    partial = encode_record(_recs(4)[3])[:-5]  # crash in the middle of a write
    with open(p, "ab") as f:
        f.write(partial)
    w2 = WriteAheadLog(p)
    out = w2.replay()
    assert len(out.records) == 3 and out.truncated_bytes == len(partial)
    assert os.path.getsize(p) == good
    w2.append([_recs(4)[3]])
    w2.close()
    assert len(WriteAheadLog(p).replay().records) == 4


def test_corrupted_byte_stops_replay(tmp_path):
    p = str(tmp_path / "w.log")
    w = WriteAheadLog(p, fsync=True)
    w.append(_recs(4))
    w.close()
    data = bytearray(open(p, "rb").read())
    frame = len(encode_record(_recs(1)[0]))
    data[frame + 20] ^= 0xFF  # flip a byte inside record 2
    open(p, "wb").write(bytes(data))
    out = WriteAheadLog(p).replay()
    assert [r.lsn for r in out.records] == [1]


def test_garbage_length_header(tmp_path):
    p = str(tmp_path / "w.log")
    with open(p, "wb") as f:
        f.write(b"\xff\xff\xff\x7f" + b"\x00" * 12)
    out = WriteAheadLog(p).replay()
    assert out.records == [] and os.path.getsize(p) == 0
