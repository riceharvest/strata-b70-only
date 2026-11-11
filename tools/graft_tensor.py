"""tools/graft_tensor.py: put one tensor of one GGUF file into another GGUF file, in place.

    python tools/graft_tensor.py --target SHARD --tensor NAME --source FILE_OR_URL [--source-sha256 HEX] [--out COPY]

Why: Unsloth's UD-IQ1_S has one layer whose gate and up experts differ in type (blk.18: gate IQ2_S, up IQ2_XXS), and
tools/iq_pack.py needs one type for both. The same repository's UD-IQ1_M has that tensor as IQ2_XXS. Nothing is
quantized again: the tensor's bytes are taken from the other file as they are.

What it does, in this order:
  1. reads both headers (the source over HTTP range requests when it is a URL: only the header and the one tensor);
  2. checks that the tensor exists in both, has the same shape, and that the source's bytes fit where the target's are;
  3. reads the source's bytes (optionally against --source-sha256) and writes them where the target's tensor was;
  4. writes zeros over the rest of the old slot, then changes the tensor's type in the target's tensor directory.
The offset of the tensor and of every other tensor stay, so the file is as long as before. The type is changed last: a run
that stops earlier leaves a file that still reads as the original, and running it again finishes the job. A tensor that is
already the source's bytes is reported and left alone. --out writes a copy and leaves the target as it is.

--compact (default off) closes the hole the graft leaves when the new tensor is smaller than the old one: every later tensor moves
down to its aligned contiguous offset, the directory offsets are rewritten and the file is cut. Strata follows the directory's
offsets and does not need it; llama.cpp refuses a file whose tensors are not contiguous. It changes the file's hash and size, and is
restartable: a journal file (<target>.compact) records the plan and the copy's progress (fsynced after each chunk); a run that
stops is finished by running the same command again. The file is not readable by anything until it is finished. It works in place
only (a copy would need the space twice); use `cp --reflink` first if you want to keep the original.

Exit status: 0 done or already done, 1 refused or failed (the target is not changed by a refusal).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import shutil
import struct
import sys
import tempfile
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gguf_reader import GGML_TYPES, GGUFFile  # noqa: E402

CHUNK = 8 << 20
HEADER_TRY = (1 << 20, 8 << 20, 64 << 20)       # how much of a remote file is read to find its tensor directory
TYPE_IDS = {name: i for i, name in GGML_TYPES.items()}


class Refused(Exception):
    """The graft cannot be done as asked; the target has not been changed."""


def is_url(src: str) -> bool:
    return str(src).startswith(("http://", "https://"))


def _http_range(url: str, start: int, length: int, tries: int = 3, allow_short: bool = False) -> bytes:
    """`length` bytes at `start`, from a 206 whose Content-Range starts where asked (a server that ignores Range sends the
    whole file from its start: that is refused, not used). A range past the end of the file comes back shorter: that is an
    error unless `allow_short`."""
    last = None
    for _ in range(tries):
        req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, start + length - 1),
                                                   "User-Agent": "strata-graft-tensor"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if r.status != 206:
                    raise Refused("%s ignored the range request (status %s)" % (url.split("/")[2], r.status))
                cr = r.headers.get("Content-Range", "")
                m = re.match(r"bytes (\d+)-(\d+)/", cr)
                if not m or int(m.group(1)) != start or int(m.group(2)) > start + length - 1:
                    raise Refused("%s answered another range than asked (%s)" % (url.split("/")[2], cr or "none"))
                data = r.read()
                length = min(length, int(m.group(2)) - start + 1) if allow_short else length
        except Refused:
            raise
        except (OSError, urllib.error.URLError) as e:
            last = e
            continue
        if len(data) != length:
            last = OSError("short read: %d of %d bytes" % (len(data), length))
            continue
        return data
    raise Refused("cannot read %s (%s)" % (url.split("/")[2], last))


def read_at(src: str, start: int, length: int) -> bytes:
    if is_url(src):
        return _http_range(src, start, length)
    with open(src, "rb") as f:
        f.seek(start)
        data = f.read(length)
    if len(data) != length:
        raise Refused("%s is too short: %d bytes wanted at %d" % (src, length, start))
    return data


def stream_at(src: str, start: int, length: int):
    """The bytes in pieces of CHUNK."""
    done = 0
    while done < length:
        n = min(CHUNK, length - done)
        yield read_at(src, start + done, n)
        done += n


def read_header(src: str) -> GGUFFile:
    """The parsed header of a local or remote GGUF file (a remote one: its first bytes, as many as the directory needs)."""
    if not is_url(src):
        try:
            return GGUFFile(pathlib.Path(src))
        except (OSError, ValueError) as e:
            raise Refused("%s: %s" % (src, e)) from None
    err = None
    for size in HEADER_TRY:
        with tempfile.TemporaryDirectory() as tmp:
            p = pathlib.Path(tmp) / "header.gguf"
            try:
                p.write_bytes(_http_range(src, 0, size, allow_short=True))
            except Refused as e:
                err = e
                if size == HEADER_TRY[0]:
                    raise
                break
            try:
                g = GGUFFile(p)
            except ValueError as e:
                err = e
                continue
            g.path = src                          # the temporary file is gone once this returns; the parse stays
            return g
    raise Refused("the tensor directory of %s could not be read (%s)" % (src, err))


def find_tensor(g: GGUFFile, name: str, what: str):
    hit = [t for t in g.tensors if t.name == name]
    if len(hit) != 1:
        raise Refused("%s has %d tensors named %s" % (what, len(hit), name))
    return hit[0]


def slot_size(g: GGUFFile, t, file_size: int) -> int:
    """The bytes between this tensor's offset and the next tensor's (or the end of the file)."""
    later = [x.offset for x in g.tensors if x.offset > t.offset]
    end = min(later) if later else file_size - g.data_start
    return end - t.offset


def type_field(target: pathlib.Path, g: GGUFFile, t) -> int:
    """Where the 4-byte type of tensor `t` sits in the target's header: its directory entry is found by name and
    checked against the parse (dims, type, offset); exactly one entry must match."""
    with open(target, "rb") as f:
        head = f.read(g.data_start)
    name = t.name.encode()
    key = struct.pack("<Q", len(name)) + name
    found, at = [], head.find(key)
    while at >= 0:
        pos = at + len(key)
        try:
            (n_dims,) = struct.unpack_from("<I", head, pos)
            dims = list(struct.unpack_from("<%dQ" % n_dims, head, pos + 4))
            tid, off = struct.unpack_from("<IQ", head, pos + 4 + 8 * n_dims)
        except struct.error:
            n_dims, dims, tid, off = -1, [], -1, -1
        if dims == t.shape and tid == t.type_id and off == t.offset:
            found.append(pos + 4 + 8 * n_dims)
        at = head.find(key, at + 1)
    if len(found) != 1:
        raise Refused("the directory entry of %s was found %d times in the header" % (t.name, len(found)))
    return found[0]


def sha256_at(src: str, start: int, length: int) -> str:
    h = hashlib.sha256()
    for piece in stream_at(src, start, length):
        h.update(piece)
    return h.hexdigest()


def graft(target, tensor: str, source: str, source_sha256: str | None = None, out=None, log=print) -> str:
    """Do the graft. Returns "grafted" or "already". Raises Refused (the target is not changed) when it cannot be done.
    `out`: a path to write a changed copy to; the target is then left as it is."""
    target = pathlib.Path(target)
    source = str(source)
    tg = GGUFFile(target) if target.is_file() else None
    if tg is None:
        raise Refused("%s is not a file" % target)
    sg = read_header(source)
    tt, st = find_tensor(tg, tensor, str(target.name)), find_tensor(sg, tensor, "the source")
    if tt.shape != st.shape:
        raise Refused("%s has shape %s here and %s in the source" % (tensor, tt.shape, st.shape))
    want = st.expected_bytes()
    have = tt.expected_bytes()
    if want is None:
        raise Refused("the source tensor is %s, whose size is not known here" % st.type_name)
    if have is None:
        raise Refused("the target tensor is %s, whose size is not known here" % tt.type_name)
    size = target.stat().st_size
    slot = slot_size(tg, tt, size)
    if want > slot:
        raise Refused("the source tensor (%s, %d bytes) is bigger than the place of the target's (%s, %d bytes)"
                      % (st.type_name, want, tt.type_name, slot))
    if tg.data_start + tt.offset + slot > size:
        raise Refused("%s ends before its last tensor does" % target.name)
    src_at = sg.data_start + st.offset
    dst_at = tg.data_start + tt.offset
    if tt.type_id == st.type_id:
        with open(target, "rb") as f:
            f.seek(dst_at)
            mine = f.read(want)
        h = hashlib.sha256(mine)
        if h.hexdigest() == sha256_at(source, src_at, want):
            log("%s is already the source's tensor (%s): nothing to do" % (tensor, st.type_name))
            return "already"
    # the source's bytes are read and checked before anything is written
    h = hashlib.sha256()
    spool = tempfile.TemporaryFile()
    try:
        for piece in stream_at(source, src_at, want):
            h.update(piece)
            spool.write(piece)
        if source_sha256 and h.hexdigest() != source_sha256.lower():
            raise Refused("the source tensor's SHA-256 is %s, not the expected %s" % (h.hexdigest(), source_sha256.lower()))
        if out is not None:
            out = pathlib.Path(out)
            shutil.copyfile(target, out)
            work = out
        else:
            work = target
        pos = type_field(work, tg, tt)
        log("%s: %s (%d bytes) -> %s (%d bytes) in a slot of %d" % (tensor, tt.type_name, have, st.type_name, want, slot))
        with open(work, "r+b") as f:
            f.seek(dst_at)
            spool.seek(0)
            while True:
                b = spool.read(CHUNK)
                if not b:
                    break
                f.write(b)
            left = slot - want
            zero = bytes(min(CHUNK, left)) if left else b""
            while left:
                n = min(len(zero), left)
                f.write(zero[:n])
                left -= n
            f.flush()
            os.fsync(f.fileno())
            f.seek(pos)                           # last: until this is written the file still reads as the original
            f.write(struct.pack("<I", st.type_id))
            f.flush()
            os.fsync(f.fileno())
    finally:
        spool.close()
    log("%s: done (tensor SHA-256 %s)" % (tensor, h.hexdigest()))
    return "grafted"


def _align(n: int, a: int) -> int:
    return (n + a - 1) // a * a


def compact_plan(g: GGUFFile, file_size: int) -> dict:
    """The tensors in file order with their new offsets (each at the aligned end of the one before), the header position of each
    offset field, and the file's new size (trailing padding after the last tensor is kept)."""
    order = sorted(g.tensors, key=lambda t: t.offset)
    rows, cursor, old_end = [], 0, 0
    for t in order:
        n = t.expected_bytes()
        if n is None:
            raise Refused("%s is %s, whose size is not known here" % (t.name, t.type_name))
        new = _align(cursor, g.alignment)
        if t.offset < new:
            raise Refused("%s overlaps the tensor before it" % t.name)
        rows.append({"name": t.name, "old": t.offset, "new": new, "size": n})
        cursor = new + n
        old_end = max(old_end, t.offset + n)
    if g.data_start + old_end > file_size:
        raise Refused("the file ends before its last tensor does")
    return {"rows": rows, "size_before": file_size, "size_after": file_size - (old_end - cursor), "data_start": g.data_start}


def compact(target, expect_size: int | None = None, log=print) -> str:
    """Close the holes between tensors in place. Returns "compacted" or "already". Restartable through <target>.compact."""
    target = pathlib.Path(target)
    journal = target.with_name(target.name + ".compact")
    if journal.exists():
        plan = json.loads(journal.read_text())
        size = target.stat().st_size
        if size not in (plan["size_before"], plan["size_after"]) and not plan["size_after"] < size < plan["size_before"]:
            raise Refused("%s does not match the journal %s; delete the journal only if you know the file is whole" % (target.name, journal.name))
        log("finishing an earlier compact run (%d of %d tensors moved)" % (plan["done"], len(plan["rows"])))
    else:
        g = GGUFFile(target)
        plan = compact_plan(g, target.stat().st_size)
        if not any(r["old"] != r["new"] for r in plan["rows"]):
            log("%s has no holes between its tensors: nothing to do" % target.name)
            return "already"
        if expect_size is not None and plan["size_after"] != expect_size:
            raise Refused("the compacted file would be %d bytes, not the expected %d" % (plan["size_after"], expect_size))
        for r, t in zip(plan["rows"], sorted(g.tensors, key=lambda t: t.offset)):
            r["pos"] = type_field(target, g, t) + 4          # the offset field follows the 4-byte type
        plan.update(done=0, chunk=0, phase="data")
        _save(journal, plan)
    ds = plan["data_start"]
    rows = plan["rows"]
    moved = sum(r["size"] for r in rows if r["old"] != r["new"])
    if plan["phase"] == "data":
        with open(target, "r+b") as f:
            for i in range(plan["done"], len(rows)):
                r = rows[i]
                if r["old"] == r["new"]:
                    plan["done"], plan["chunk"] = i + 1, 0
                    continue
                done = plan["chunk"] if i == plan["done"] else 0
                while done < r["size"]:
                    # a chunk no longer than the distance moved never overwrites its own source, so a chunk written but not
                    # yet journaled is simply copied again
                    n = min(CHUNK, r["old"] - r["new"], r["size"] - done)
                    f.seek(ds + r["old"] + done)
                    b = f.read(n)
                    f.seek(ds + r["new"] + done)
                    f.write(b)
                    f.flush()
                    os.fsync(f.fileno())
                    done += n
                    plan["done"], plan["chunk"] = i, done          # a chunk ahead of this one is never touched by the write
                    _save(journal, plan)
                plan["done"], plan["chunk"] = i + 1, 0
            plan["phase"] = "header"
            _save(journal, plan)
    with open(target, "r+b") as f:
        for r in rows:
            f.seek(r["pos"])
            f.write(struct.pack("<Q", r["new"]))
        f.truncate(plan["size_after"])
        f.flush()
        os.fsync(f.fileno())
    journal.unlink()
    log("%d tensors moved (%d bytes), the file is %d bytes now" % (sum(r["old"] != r["new"] for r in rows), moved, plan["size_after"]))
    return "compacted"


def _save(journal: pathlib.Path, plan: dict) -> None:
    tmp = journal.with_name(journal.name + ".tmp")
    with open(tmp, "w") as f:
        json.dump(plan, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, journal)


def grafted(target, tensor: str, type_name: str) -> bool:
    """Does the target's tensor already have this type (the cheap test of an earlier run)?"""
    t = [x for x in GGUFFile(pathlib.Path(target)).tensors if x.name == tensor]
    return len(t) == 1 and t[0].type_name == type_name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="the GGUF file to change (one shard)")
    ap.add_argument("--tensor", help="the tensor's name, e.g. blk.18.ffn_gate_exps.weight")
    ap.add_argument("--source", help="the GGUF file (or https URL) that holds the tensor to take")
    ap.add_argument("--compact", action="store_true", help="afterwards (or alone, without --tensor) close the holes between tensors, in place")
    ap.add_argument("--expect-size", type=int, help="with --compact: the file size the result must have")
    ap.add_argument("--source-sha256", help="the SHA-256 the source tensor's bytes must have")
    ap.add_argument("--out", help="write a changed copy here and leave --target as it is")
    a = ap.parse_args(argv)
    if bool(a.tensor) != bool(a.source) or not (a.tensor or a.compact):
        ap.error("give --tensor and --source (a graft), or --compact alone, or both")
    if a.compact and a.out:
        ap.error("--compact works in place; copy the file first (cp --reflink)")
    try:
        if a.tensor:
            graft(a.target, a.tensor, a.source, a.source_sha256, a.out)
        if a.compact:
            compact(a.target, a.expect_size)
    except Refused as e:
        print("graft_tensor: refused: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
