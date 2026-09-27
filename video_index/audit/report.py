"""Report stage: one report card per benchmark (Markdown) and one summary table (CSV), from the tables of the
earlier stages. Outputs: tables/report/<bench>.md, tables/report_cards.csv."""
from __future__ import annotations

import csv
import os

from .items import select

LEVEL_TEXT = {"option": "an attacker given the options only", "text": "an attacker given the question and the options",
              "pool": "an attacker trained on the other items of the benchmark", "frame": "an attacker given one frame "
              "or frame captions", "order": "an attacker given shuffled frames or one tenth of the video",
              "unbroken": "no attacker of the five levels"}


def _table(ctx, name, key="benchmark"):
    p = os.path.join(ctx.work_dir, "tables", name)
    out = {}
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            out.setdefault(r[key], []).append(r)
    return out


def _pct(v):
    try:
        return f"{100 * float(v):.1f}"
    except (TypeError, ValueError):
        return "n/a"


def card(bench, lv, attackers, screen, perm, pool):
    L = [f"# {bench}", ""]
    if not lv:
        return "\n".join(L + ["No row in tables/levels.csv; run the levels stage.", ""])
    L += ["| Quantity | Value |", "|---|---|",
          f"| Items in the sample (after the screen) | {lv['n_items']} |",
          f"| Multiple-choice items | {lv['n_mcq']} |",
          f"| Chance c (%) | {_pct(lv.get('c'))} |",
          f"| Reference model | {lv.get('reference_model') or 'n/a'} |",
          f"| Reference accuracy s* at 32 frames (%) | {_pct(lv.get('s_star'))} (n = {lv.get('n_ref')}) |",
          f"| Threshold over chance (%) | {_pct(lv.get('threshold'))} |",
          f"| Breaking level (delta = {lv.get('delta') or 0.05}) | {lv.get('break_level') or 'none'} |",
          f"| Breaking level at delta = 0.03 / 0.08 | {lv.get('break_level_delta3') or 'n/a'} / {lv.get('break_level_delta8') or 'n/a'} |"]
    if lv.get("second_reference_break_level"):
        L.append(f"| Breaking level with the second reference ({lv['sr_model']}) | {lv['second_reference_break_level']} |")
    if lv.get("rescreened_128") in ("1", 1):
        L.append(f"| 128-frame reference accuracy used as s* (%) | {_pct(lv.get('s_star_128'))} |")
    if lv.get("reference_near_chance") in ("1", 1):
        L.append("| Reference within delta of chance | yes |")
    if screen:
        L.append(f"| Near-duplicates removed / option counts repaired | {screen['n_duplicate_removed']} / {screen['n_option_count_repaired']} |")
    if perm and perm.get("width"):
        L.append(f"| Option permutations: accuracy width (%) / threshold (%) / item consistency (%) | "
                 f"{_pct(perm['width'])} / {_pct(perm['threshold'])} / {_pct(perm['item_consistency'])} |")
    if lv.get("break_level"):
        L += ["", f"The reference margin over chance is matched within delta by {LEVEL_TEXT.get(lv['break_level'], lv['break_level'])}."
              if lv["break_level"] != "unbroken" else "", ]
    L += ["", "## Margin over chance per level (%)", "", "| Level | Largest margin (running maximum) |", "|---|---|"]
    for l, col in (("option", "eps_opt"), ("text", "eps_text"), ("pool", "eps_pool"), ("frame", "eps_frame"), ("order", "eps_order")):
        L.append(f"| {l} | {_pct(lv.get(col)) if lv.get(col) else 'not measured'} |")
    if attackers:
        L += ["", "## Attackers", "", "| Level | Attacker | Accuracy (%) | Margin (%) | n |", "|---|---|---|---|---|"]
        L += [f"| {a['level']} | {a['attacker']} | {_pct(a['acc'])} | {_pct(a['margin'])} | {a['n']} |" for a in attackers]
    if lv.get("notes"):
        L += ["", "## Notes", "", lv["notes"]]
    return "\n".join(L) + "\n"


def run(ctx, benchmarks=None, limit=None):
    lv, att = _table(ctx, "levels.csv"), _table(ctx, "level_attackers.csv")
    scr, perm, pool = _table(ctx, "screen_counts.csv"), _table(ctx, "permutation_by_benchmark.csv"), _table(ctx, "pool_attack.csv")
    out = []
    for bench in select(ctx, benchmarks):
        l = (lv.get(bench) or [None])[0]
        p = ctx.path("table", name=f"report/{bench}.md")
        with open(p, "w") as f:
            f.write(card(bench, l, att.get(bench, []), (scr.get(bench) or [None])[0], (perm.get(bench) or [None])[0],
                         pool.get(bench, [])))
        if l:
            pm = (perm.get(bench) or [{}])[0]
            out.append(dict(benchmark=bench, n_mcq=l["n_mcq"], c=l.get("c", ""), s_star=l.get("s_star", ""),
                            reference_model=l.get("reference_model", ""), eps_opt=l.get("eps_opt", ""),
                            eps_text=l.get("eps_text", ""), eps_pool=l.get("eps_pool", ""), eps_frame=l.get("eps_frame", ""),
                            eps_order=l.get("eps_order", ""), break_level=l.get("break_level", ""),
                            permutation_width=pm.get("width", ""), report=os.path.relpath(p, ctx.work_dir)))
    if out:
        with open(ctx.path("table", name="report_cards.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out[0]))
            w.writeheader()
            w.writerows(out)
    return out
