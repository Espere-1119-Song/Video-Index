"""Sources, format check, sampling and the command line on a small local benchmark."""
import json
import os

from video_index.data.schema import read_jsonl, write_jsonl
from video_index.onboard import cli, formatcheck, registry, sources
from video_index.onboard.adapters import to_item


def make_benchmark(root, n=60, yes_no=5, open_items=4):
    os.makedirs(root / "ann", exist_ok=True)
    os.makedirs(root / "videos" / "clips", exist_ok=True)
    rows = []
    for i in range(n):
        (root / "videos" / "clips" / f"v{i:03d}.mp4").write_bytes(b"\x00" * 16)
        rows.append(dict(video=f"v{i:03d}", question=f"What happens in clip {i}?", options=["A. one", "B. two", "C. three", "D. four"],
                         answer="ABCD"[i % 4], task=["count", "order", "state"][i % 3]))
    rows += [dict(video="v000", question="Is it raining?", options=["Yes", "No"], answer="Yes", task="state") for _ in range(yes_no)]
    rows += [dict(video="v001", question=f"Describe clip {k}.", answer="A long free-text description.", task="state") for k in range(open_items)]
    rows.append(dict(video="missing_video", question="Q?", options=["A. x", "B. y"], answer="A", task="count"))
    (root / "ann" / "test.json").write_text(json.dumps(rows))
    return str(root / "ann"), str(root / "videos")


def test_local_video_index_and_fetch(tmp_path):
    _, vids = make_benchmark(tmp_path, n=3, yes_no=0, open_items=0)
    src = sources.parse_source(vids)
    assert src == {"kind": "local", "root": vids}
    idx = sources.VideoIndex(src, str(tmp_path / "cache")).build()
    ref = idx.resolve("v001")                                   # stem -> file
    assert ref["member"] == "clips/v001.mp4" and idx.resolve("clips/v002.mp4") and idx.resolve("nope") is None
    assert idx.resolve(("a.zip", "x/y.mp4")) == dict(source=sources.label(src), archive="a.zip", member="x/y.mp4")
    out = sources.fetch_video(src, ref, str(tmp_path / "out" / "v.mp4"))
    assert os.path.getsize(out) == 16
    assert os.path.exists(idx.cache) and sources.VideoIndex(src, str(tmp_path / "cache")).build().idx == idx.idx
    assert sources.parse_source("owner/name") == {"kind": "hf", "repo": "owner/name"}
    assert sources.parse_source("urls::list.tsv") == {"kind": "urls", "list": "list.tsv"}


def test_custom_fetcher(tmp_path):
    @sources.register_fetcher("clips")
    def fetch(source, ref, dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "wb").write(ref["member"].encode())
        return dest
    src = {"kind": "clips", "target": "x"}
    ref = sources.VideoIndex(src).build().resolve("abc|10-20")
    assert ref["member"] == "abc|10-20"
    done, failed = sources.fetch_videos(src, [ref], lambda r: str(tmp_path / "c.mp4"), log=lambda m: None)
    assert not failed and open(done[sources.ref_key(ref)], "rb").read() == b"abc|10-20"


def test_format_report_and_eligibility():
    items = [to_item("B", i, dict(v=f"v{i}", q=f"What is shown in clip {i}?", a="ABCD"[i % 4], opts=["w", "x", "y", "z"], sub="s"))
             for i in range(40)]
    items += [to_item("B", 100, dict(v="v", q="Is it on?", a="Yes", opts=["Yes", "No"], sub="s"))]
    rep = formatcheck.format_report(items)
    assert (rep["n_mcq"], rep["n_yesno"], rep["chance"], rep["position_entropy"]) == (40, 1, 0.25, 1.0) and rep["eligible"]
    few = formatcheck.format_report(items[:10])
    assert not few["eligible"] and [c["rule"] for c in few["checks"] if not c["passed"]] == ["min_items"]
    audio = formatcheck.format_report(items, requires_audio=True)
    assert [c["rule"] for c in audio["checks"] if not c["passed"]] == ["no_audio"]
    zh = [to_item("B", i, dict(v="v", q="视频中的人在做什么事情?", a="A", opts=["跑步", "游泳"], sub="s")) for i in range(40)]
    assert [c["rule"] for c in formatcheck.format_report(zh)["checks"] if not c["passed"]] == ["english"]
    no_gold = [to_item("B", i, dict(v="v", q="What is shown?", a="", opts=["a", "b"], sub="s")) for i in range(40)]
    failed = [c["rule"] for c in formatcheck.format_report(no_gold)["checks"] if not c["passed"]]
    assert "gold_available" in failed and "multiple_choice" in failed
    skew = [to_item("B", i, dict(v="v", q="What is shown?", a="A", opts=["a", "b", "c"], sub="s")) for i in range(40)]
    assert formatcheck.format_report(skew)["position_entropy"] == 0.0


def test_stratified_sample_is_seeded_and_proportional():
    items = [dict(qid=f"B_{i}", subtask="big" if i < 900 else "small") for i in range(1000)]
    a, full = registry.stratified_sample(items, 300, seed=42)
    b, _ = registry.stratified_sample(list(reversed(items)), 300, seed=42)
    assert not full and len(a) == 300 and [r["qid"] for r in a] == [r["qid"] for r in b]
    assert sum(r["subtask"] == "small" for r in a) == 30
    tiny = [dict(qid=f"B_{i}", subtask="big" if i < 990 else "rare") for i in range(1000)]
    assert sum(r["subtask"] == "rare" for r in registry.stratified_sample(tiny, 300)[0]) == 10      # floor per subtask
    assert registry.stratified_sample(items[:50], 300) == (sorted(items[:50], key=lambda r: r["qid"]), True)


def test_check_and_add_on_a_local_benchmark(tmp_path, capsys, monkeypatch):
    ann, vids = make_benchmark(tmp_path / "src")
    work = str(tmp_path / "work")
    monkeypatch.chdir(tmp_path)                                  # no configs/ here: detection, default settings
    rc = cli.main(["check", "--name", "Tiny/Bench", "--repo", ann, "--video-repo", vids, "--work-dir", work, "--resolve-videos"])
    out = capsys.readouterr().out
    assert rc == 0 and "multiple choice 60, open 4, yes/no 5" in out and "=> eligible" in out
    monkeypatch.setattr(cli, "run_audit_stages", fallback_sample)
    rc = cli.main(["add", "--name", "Tiny/Bench", "--repo", ann, "--video-repo", vids, "--work-dir", work, "--sample-n", "30",
                   "--stages", "sample,fetch"])
    assert rc == 0
    items = read_jsonl(os.path.join(work, "items", "Tiny_Bench.jsonl"))
    sample = read_jsonl(os.path.join(work, "samples", "Tiny_Bench.jsonl"))
    assert len(items) == 60 and all(i["format"] == "mcq" for i in items) and len(sample) == 30
    assert all(r["video_path"] and os.path.exists(os.path.join(work, r["video_path"])) for r in sample)
    assert len(read_jsonl(os.path.join(work, "tables", "format", "Tiny_Bench_excluded.jsonl"))) == 9
    reg = registry.read_csv(os.path.join(work, "tables", "registry.csv"))
    assert reg[0]["name"] == "Tiny/Bench" and reg[0]["n_mcq"] == "60" and reg[0]["n_sampled"] == "30" and reg[0]["status"] == "onboarded"
    assert cli.main(["list", "--work-dir", work]) == 0 and "Tiny/Bench" in capsys.readouterr().out
    # a benchmark that fails a rule is registered as not eligible and nothing is sampled
    ann2, vids2 = make_benchmark(tmp_path / "small", n=12)
    rc = cli.main(["add", "--name", "Small", "--repo", ann2, "--video-repo", vids2, "--work-dir", work, "--stages", "sample"])
    assert rc == 1 and not os.path.exists(os.path.join(work, "samples", "Small.jsonl"))
    assert {r["name"]: r["status"] for r in registry.read_csv(os.path.join(work, "tables", "registry.csv"))}["Small"] == "not_eligible"


def fallback_sample(ctx, name, stages, limit):
    """Stands in for video_index.audit.cli.run_stages: the sample stage of the onboarding package."""
    assert stages == ["sample"]
    items = read_jsonl(ctx.path("items", bench=cli.safe_name(name)))
    write_jsonl(ctx.path("samples", bench=cli.safe_name(name)), registry.stratified_sample(items, int(ctx.get("sample_n")))[0])
    return True
