"""Where annotations and videos come from.

A source is a small dict (in an adapter spec under ``source.annotations`` / ``source.videos``, or built from the
command line):

    {kind: hf, repo: owner/name, revision: <sha or tag>}     a Hugging Face dataset repository
    {kind: hf, repo: owner/name, include: ["videos_*.zip"]}  the same, restricted to matching files or archives
    {kind: local, root: /path/to/dir}                        a directory on this machine
    {kind: urls, list: videos.tsv}                           a list of "name<TAB>url" lines (or a JSON object)
    {kind: <custom>, ...}                                    a fetcher added with :func:`register_fetcher`

Videos inside archives are fetched member by member: a zip is read through HTTP range requests (only the member's
bytes are transferred), a plain tar likewise, a split tar (``x.tar.part*``) is joined on the fly, and a compressed tar
is streamed once for all wanted members. Nothing is downloaded that the sample does not need.

YouTube and other sites that do not redistribute files are not supported by default. Add a fetcher for them:

    from video_index.onboard.sources import register_fetcher
    @register_fetcher("youtube")
    def fetch(source, ref, dest): ...        # write the video to dest, return dest

A Hugging Face token is read from ``HF_TOKEN`` or from the login stored by ``huggingface-cli login``.
"""
from __future__ import annotations

import fnmatch
import json
import os
import re
import shutil
import tarfile
import zipfile
from typing import Callable

VIDEO_EXT = (".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v")
ANNOTATION_EXT = (".json", ".jsonl", ".csv", ".tsv", ".parquet")
TAR_EXT = (".tar", ".tar.gz", ".tgz", ".tar.bz2")
TAR_COMPRESSED = (".tar.gz", ".tgz", ".tar.bz2")
TAR_STREAM_CAP_GB = 10           # a compressed tar has no random access: larger ones are not indexed
ANNOTATION_CAP_GB = 2.0          # annotation files above this size are skipped (they are media dumps)

FETCHERS: dict[str, Callable] = {}


class SourceError(RuntimeError):
    pass


def register_fetcher(kind: str):
    """Decorator for a video fetcher of a custom source kind: ``fn(source: dict, ref: dict, dest: str) -> dest``."""
    def deco(fn):
        FETCHERS[kind] = fn
        return fn
    return deco


def parse_source(text: str | dict | None) -> dict | None:
    """``owner/name`` -> hf, an existing directory -> local, ``hf::x`` / ``local::x`` / ``urls::x`` explicit."""
    if text is None or isinstance(text, dict):
        return text
    if "::" in text:
        kind, val = text.split("::", 1)
        key = {"hf": "repo", "local": "root", "urls": "list"}.get(kind, "target")
        return {"kind": kind, key: val}
    if os.path.isdir(text):
        return {"kind": "local", "root": os.path.abspath(text)}
    return {"kind": "hf", "repo": text}


def label(source: dict) -> str:
    return f"{source['kind']}::{source.get('repo') or source.get('root') or source.get('list') or source.get('target', '')}"


# ------------------------------------------------------------------ Hugging Face helpers
def _hf_token():
    return os.environ.get("HF_TOKEN") or None        # None: huggingface_hub falls back to the stored login


_fs = None


def _hffs():
    global _fs
    if _fs is None:
        from huggingface_hub import HfFileSystem
        _fs = HfFileSystem(token=_hf_token())
    return _fs


def hf_tree(repo: str, revision: str | None = None) -> list[dict]:
    """Every file of the repository with its size (paginated listing; the plain tree endpoint stops at 1,000 entries)."""
    from huggingface_hub import HfApi
    out = []
    for it in HfApi(token=_hf_token()).list_repo_tree(repo, repo_type="dataset", recursive=True, expand=False,
                                                      revision=revision):
        if getattr(it, "size", None) is not None:
            out.append(dict(path=it.path, size=it.size))
    return out


def _natural(p):
    """'x.part2' before 'x.part10': a lexicographic order corrupts a joined split archive."""
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(p))]


class ConcatFile:
    """Read-only seekable concatenation of remote files (split ``.tar.part*`` archives)."""

    def __init__(self, fs, paths, sizes=None):
        self.fs, self.parts, off = fs, [], 0
        for i, p in enumerate(paths):
            size = sizes[i] if sizes and sizes[i] is not None else fs.info(p)["size"]
            self.parts.append((off, size, p))
            off += size
        self.size, self.pos, self._open = off, 0, {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def seek(self, pos, whence=0):
        if whence == 1:
            pos += self.pos
        elif whence == 2:
            pos += self.size
        self.pos = max(0, pos)
        return self.pos

    def tell(self):
        return self.pos

    def seekable(self):
        return True

    def readable(self):
        return True

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        out = []
        while n > 0 and self.pos < self.size:
            hit = next(((o, s, p) for o, s, p in self.parts if o <= self.pos < o + s), None)
            if hit is None:
                break
            off, size, p = hit
            f = self._open.get(p)
            if f is None:
                f = self._open[p] = self.fs.open(p, "rb")
            f.seek(self.pos - off)
            chunk = f.read(min(n, off + size - self.pos))
            if not chunk:
                break
            out.append(chunk)
            self.pos += len(chunk)
            n -= len(chunk)
        return b"".join(out)

    def close(self):
        for f in self._open.values():
            f.close()


_zip_handles: dict = {}
_tar_handles: dict = {}
_glob_cache: dict = {}


def _open_remote(key: str):
    """A handle for a repository path; a ``*`` in the path means split chunks joined in natural order."""
    fs = _hffs()
    if "*" in key:
        cached = _glob_cache.get(key)
        if cached is None:
            detail = fs.glob(key, detail=True)
            if not detail:
                raise FileNotFoundError(key)
            cached = sorted(((p, (info or {}).get("size")) for p, info in detail.items()), key=lambda x: _natural(x[0]))
            _glob_cache[key] = cached
        return ConcatFile(fs, [p for p, _ in cached], [s for _, s in cached])
    return fs.open(key, "rb")


def remote_zip(repo: str, path: str) -> zipfile.ZipFile:
    key = f"datasets/{repo}/{path}"
    if key not in _zip_handles:
        _zip_handles[key] = zipfile.ZipFile(_hffs().open(key, "rb"))
    return _zip_handles[key]


def remote_tar(repo: str, path: str):
    """Seekable tarfile for a plain tar (also a split one); None for a compressed tar."""
    key = f"datasets/{repo}/{path}"
    if key not in _tar_handles:
        if path.lower().endswith(TAR_COMPRESSED) or "*" in path:
            _tar_handles[key] = None
        else:
            fs = _hffs()
            if fs.exists(key):
                f = fs.open(key, "rb")
            else:
                parts = sorted(fs.glob(key + ".part*"), key=_natural)
                if not parts:
                    raise FileNotFoundError(key)
                f = ConcatFile(fs, parts)
            _tar_handles[key] = tarfile.open(fileobj=f, mode="r:")
    return _tar_handles[key]


def _tar_names(repo: str, path: str) -> list[str]:
    tf = remote_tar(repo, path)
    if tf is not None:
        return [m.name for m in tf.getmembers() if m.isfile()]
    names = []
    with _open_remote(f"datasets/{repo}/{path}") as f, tarfile.open(fileobj=f, mode="r|*") as t:
        for m in t:
            if m.isfile():
                names.append(m.name)
    return names


def _copy(src, dest):
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    tmp = dest + ".part"
    with open(tmp, "wb") as out:
        shutil.copyfileobj(src, out, 1 << 20)
    os.replace(tmp, dest)
    return dest


# ------------------------------------------------------------------ annotations
def fetch_annotations(source: dict, dest_dir: str, include: list[str] | None = None) -> list[str]:
    """Bring the annotation files of the source below ``dest_dir``; returns the relative paths."""
    os.makedirs(dest_dir, exist_ok=True)
    include = include or source.get("include")
    kind = source["kind"]

    def wanted(rel: str, size: float) -> bool:
        if not rel.lower().endswith(ANNOTATION_EXT) or size > ANNOTATION_CAP_GB * 1e9:
            return False
        return not include or any(fnmatch.fnmatch(rel, p) or p in rel for p in include)

    got = []
    if kind == "hf":
        from huggingface_hub import hf_hub_download
        for it in hf_tree(source["repo"], source.get("revision")):
            if wanted(it["path"], it["size"]):
                hf_hub_download(repo_id=source["repo"], filename=it["path"], repo_type="dataset", local_dir=dest_dir,
                                token=_hf_token(), revision=source.get("revision"))
                got.append(it["path"])
    elif kind == "local":
        root = source["root"]
        for d, _, files in os.walk(root):
            for f in files:
                p = os.path.join(d, f)
                rel = os.path.relpath(p, root)
                if "/." in "/" + rel or not wanted(rel, os.path.getsize(p)):
                    continue
                out = os.path.join(dest_dir, rel)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                if not os.path.exists(out):
                    try:
                        os.symlink(os.path.abspath(p), out)
                    except OSError:
                        shutil.copyfile(p, out)
                got.append(rel)
    elif kind == "urls":
        import requests
        for name, url in _url_list(source["list"]).items():
            if name.lower().endswith(ANNOTATION_EXT):
                r = requests.get(url, timeout=300)
                r.raise_for_status()
                out = os.path.join(dest_dir, name)
                os.makedirs(os.path.dirname(out), exist_ok=True)
                open(out, "wb").write(r.content)
                got.append(name)
    else:
        raise SourceError(f"annotations: unsupported source kind '{kind}'")
    if not got:
        raise SourceError(f"no annotation file ({', '.join(ANNOTATION_EXT)}) found in {label(source)}")
    return sorted(got)


def _url_list(path: str) -> dict[str, str]:
    if path.lower().endswith(".json"):
        return dict(json.load(open(path)))
    out = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            parts = re.split(r"\t|,|\s+", line, maxsplit=1)
            if len(parts) == 2:
                out[parts[0]] = parts[1].strip()
            else:
                out[os.path.basename(parts[0].split("?")[0])] = parts[0]
    return out


# ------------------------------------------------------------------ video index
class VideoIndex:
    """Every video of a source under three keys (full path, file name, stem) -> ``[archive, member]``.
    Direct files win over archive members; the index is cached as JSON."""

    def __init__(self, source: dict, cache_dir: str | None = None):
        self.source = source
        tag = label(source) + ("::" + ",".join(source["include"]) if source.get("include") else "")
        self.cache = os.path.join(cache_dir, re.sub(r"[^A-Za-z0-9_.-]+", "__", tag) + ".json") if cache_dir else None
        self.idx: dict[str, list] = {}
        self.skipped: list[str] = []

    def build(self) -> "VideoIndex":
        if self.cache and os.path.exists(self.cache):
            self.idx = json.load(open(self.cache))
            return self
        kind = self.source["kind"]
        if kind == "hf":
            self._build_hf()
        elif kind == "local":
            root = self.source["root"]
            for d, _, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(VIDEO_EXT):
                        self._add("", os.path.relpath(os.path.join(d, f), root), direct=True)
        elif kind == "urls":
            for name in _url_list(self.source["list"]):
                self._add("", name, direct=True)
        else:
            return self                       # custom kinds resolve nothing: references are passed through
        if self.cache and self.idx:
            os.makedirs(os.path.dirname(self.cache), exist_ok=True)
            json.dump(self.idx, open(self.cache + ".tmp", "w"))
            os.replace(self.cache + ".tmp", self.cache)
        return self

    def _add(self, archive: str, member: str, direct: bool = False):
        hit = [archive, member]
        base = os.path.basename(member)
        for k in (member, base, os.path.splitext(base)[0]):
            if direct and k == member:
                self.idx[k] = hit
            else:
                self.idx.setdefault(k, hit)

    def _build_hf(self):
        repo, rev = self.source["repo"], self.source.get("revision")
        files = hf_tree(repo, rev)
        include = self.source.get("include")                  # file or archive patterns, e.g. ["video_chunk_*.zip"]
        if include:
            files = [it for it in files if any(fnmatch.fnmatch(it["path"], p) or p in it["path"] for p in include)]
        part = re.compile(r"^(.+\.tar)\.part[.\-_]?[A-Za-z0-9]*$", re.I)
        direct, zips, tars, bases = [], [], [], set()
        for it in files:
            p = it["path"]
            m = part.match(p)
            if m:
                bases.add(m.group(1))
            elif p.lower().endswith(VIDEO_EXT):
                direct.append(p)
            elif p.lower().endswith(".zip"):
                zips.append(p)
            elif p.lower().endswith(TAR_EXT):
                if p.lower().endswith(TAR_COMPRESSED) and it["size"] > TAR_STREAM_CAP_GB * 1e9:
                    self.skipped.append(f"{p}: compressed tar of {it['size'] / 1e9:.1f} GB above the stream cap")
                else:
                    tars.append(p)
        tars.extend(b for b in sorted(bases) if b not in tars)
        for zp in zips:
            try:
                for m in remote_zip(repo, zp).namelist():
                    if m.lower().endswith(VIDEO_EXT):
                        self._add(zp, m)
            except Exception as e:  # noqa: BLE001
                _zip_handles.pop(f"datasets/{repo}/{zp}", None)
                self.skipped.append(f"{zp}: {type(e).__name__}: {str(e)[:120]}")
        for tp in tars:
            try:
                for m in _tar_names(repo, tp):
                    if m.lower().endswith(VIDEO_EXT):
                        self._add(tp, m)
            except Exception as e:  # noqa: BLE001
                self.skipped.append(f"{tp}: {type(e).__name__}: {str(e)[:120]}")
        for p in direct:
            self._add("", p, direct=True)

    def resolve(self, v) -> dict | None:
        """The adapter's video reference -> ``{source, archive, member}``, or None when nothing matches.
        A pinned ``(archive, member)`` pair is taken as given."""
        if v is None:
            return None
        src = label(self.source)
        if isinstance(v, (tuple, list)):
            if len(v) == 3:                                   # (source label, archive, member): items from several sources
                return dict(source=v[0] or src, archive=v[1], member=v[2])
            if len(v) == 2:
                return dict(source=src, archive=v[0], member=v[1])
        v = str(v).strip().strip("/")
        if not v:
            return None
        if self.source["kind"] not in ("hf", "local", "urls"):
            return dict(source=src, archive="", member=v)
        base = os.path.basename(v)
        hit = (self.idx.get(v) or self.idx.get(base) or self.idx.get(os.path.splitext(base)[0])
               or self.idx.get(base + ".mp4") or self.idx.get(v + ".mp4"))
        return dict(source=src, archive=hit[0], member=hit[1]) if hit else None


def ref_key(ref: dict | str) -> str:
    if isinstance(ref, str):
        return ref
    return f"{ref.get('source', '')}::{ref.get('archive', '')}::{ref.get('member', '')}"


# ------------------------------------------------------------------ videos
def fetch_video(source: dict, ref: dict, dest: str) -> str:
    """Write one video to ``dest``."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest
    kind = source["kind"]
    archive, member = ref.get("archive") or "", ref["member"]
    if kind in FETCHERS:
        return FETCHERS[kind](source, ref, dest)
    if kind == "local":
        p = os.path.join(source["root"], member)
        if not os.path.exists(p):
            raise SourceError(f"missing file {p}")
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        try:
            os.symlink(os.path.abspath(p), dest)
        except OSError:
            shutil.copyfile(p, dest)
        return dest
    if kind == "urls":
        import requests
        url = _url_list(source["list"])[member]
        with requests.get(url, stream=True, timeout=600) as r:
            r.raise_for_status()
            return _copy(r.raw, dest)
    if kind != "hf":
        raise SourceError(f"no fetcher for source kind '{kind}' (add one with register_fetcher)")
    repo = source["repo"]
    if not archive:
        with _hffs().open(f"datasets/{repo}/{member}", "rb", revision=source.get("revision")) as f:
            return _copy(f, dest)
    if archive.lower().endswith(".zip"):
        with remote_zip(repo, archive).open(member) as f:
            return _copy(f, dest)
    tf = remote_tar(repo, archive)
    if tf is not None:
        return _copy(tf.extractfile(member), dest)
    got = fetch_tar_members(repo, archive, {member: dest})
    if member not in got:
        raise SourceError(f"{member} not in {archive}")
    return dest


def fetch_tar_members(repo: str, archive: str, wanted: dict[str, str]) -> dict[str, str]:
    """One streaming pass through a compressed tar for every wanted member ({member: dest})."""
    got = {}
    with _open_remote(f"datasets/{repo}/{archive}") as f, tarfile.open(fileobj=f, mode="r|*") as t:
        for m in t:
            if m.name in wanted:
                _copy(t.extractfile(m), wanted[m.name])
                got[m.name] = wanted[m.name]
                if len(got) == len(wanted):
                    break
    return got


def fetch_videos(source: dict, refs: list[dict], dest_of: Callable[[dict], str], log=print) -> tuple[dict, dict]:
    """Fetch every reference; members of one compressed tar share a single pass. Returns ({key: path}, {key: error})."""
    done, failed, streamed = {}, {}, {}
    for ref in refs:
        a = ref.get("archive") or ""
        if source["kind"] == "hf" and (a.lower().endswith(TAR_COMPRESSED) or "*" in a):
            streamed.setdefault(a, []).append(ref)
    for a, rs in streamed.items():
        want = {r["member"]: dest_of(r) for r in rs if not os.path.exists(dest_of(r))}
        try:
            if want:
                fetch_tar_members(source["repo"], a, want)
        except Exception as e:  # noqa: BLE001
            log(f"archive {a}: {type(e).__name__}: {str(e)[:160]}")
    for k, ref in enumerate(refs, 1):
        dest = dest_of(ref)
        try:
            done[ref_key(ref)] = dest if os.path.exists(dest) else fetch_video(source, ref, dest)
        except Exception as e:  # noqa: BLE001
            failed[ref_key(ref)] = f"{type(e).__name__}: {str(e)[:200]}"
        if k % 25 == 0:
            log(f"videos: {k}/{len(refs)} ({len(failed)} failed)")
    return done, failed
