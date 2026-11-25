"""Agreement between TRACE-LOCO's MLM and TRACE-FarmCPU in the simulations, counted for signals and
for exact SNPs. Counting only: no GWAS run. The saved results of the 9 scenarios x 100 replicates are
scored with the function that scored Tables S2 and S12, classify_detections in
benchmarks/simulation/evaluate_simulation.py (this folder), imported unchanged.

Inputs, in SIM (default benchmarks/simulation/sim_data, not in the repository):
- MLM: <scenario>/rep_<k>/platform_loco/results.csv, the TRACE-LOCO rows of Table S2;
- FarmCPU: <scenario>/rep_<k>/farmcpu_2x2_E_F_nogate2_ols/results.csv, the TRACE-FarmCPU rows of
  Table S12;
- truth: <scenario>/rep_<k>/truth.json; positions: snp_map.csv.

Rules, fixed before the run:
- Significant: p < 0.05 / 43,974, each model on its own p-values. MLM's results are read as
  evaluate_all_scenarios reads them (rows without a p-value dropped), FarmCPU's as saved.
- Within 500 kb: on the same chromosome and at most 500,000 bp apart, as classify_detections counts
  a true SNP. The clumping and the agreement use the same distance.
- Signals (main analysis), per replicate and per model: sort the significant SNPs by p, ties by
  chromosome, then position, then SNP ID. The most significant SNP not yet used leads a signal, and
  every unused significant SNP within 500 kb of the lead joins it; repeat until every SNP is used.
  A signal is represented by its lead SNP, and it is true when a causal QTN is within 500 kb of the
  lead.
- Agreement: an MLM signal and a FarmCPU signal agree when their lead SNPs are within 500 kb. The
  groups: MLM signals with an agreeing FarmCPU signal, MLM signals without one, FarmCPU signals with
  an agreeing MLM signal, FarmCPU signals without one.
- Exact SNPs (secondary analysis, TRACE's consensus rule, gwas/pipeline.py:712-742): every
  significant SNP is in one class: both (the same SNP ID is significant in both models), MLM only or
  FarmCPU only. A SNP is true when a causal QTN is within 500 kb.
- Per scenario: each count as the total over the replicates and the median per replicate (a
  replicate with none counts 0); the number true; precision = true / all from the totals, empty when
  the count is 0.
- Recall, computed as power is for Table S2: per replicate, the share of the causal QTNs with a lead
  SNP (signals) or a SNP (exact SNPs) of the set within 500 kb, averaged over the replicates. The
  sets: the agreeing signals (the leads of both agreeing groups together) or the both class; all of
  MLM's; all of FarmCPU's; either model (all of both models').
- Pooled over the 9 scenarios: the totals, the median over the 900 replicates, precision from the
  totals and recall averaged over the 900 replicates.
- The number true, recall and power always come from classify_detections on the rows concerned.
- In the tables (consensus_tables.md), percentages are rounded half up from the exact counts.

Checks (a failed one stops the script before the consensus files are written; facts.json says which):
1. MLM only: per replicate, its power, FDR and number of significant SNPs equal the committed
   scores in sim_summary/bonferroni_results.csv (platform_loco; power and FDR to 1e-15, as that file
   holds at most 16 significant digits), and summarize_with_ci over these scores gives the committed
   TRACE-LOCO rows of sim_summary/power_fdr_summary_ci.csv (Table S2): n_reps, and power and FDR
   with their SE and CI, as written;
2. per replicate, every significant SNP is in exactly one signal of its model; the agreeing and the
   not agreeing signals add up to the model's signals; both plus MLM only is MLM's number of
   significant SNPs, and both plus FarmCPU only is FarmCPU's;
and the guards: 43,974 markers; distinct SNP IDs among each model's significant SNPs; their Chr and
Pos equal snp_map.csv's; the true counts add up as the counts do.

    python benchmarks/simulation/consensus_precision.py --out DIR [--sim-data SIM]

Writes, in DIR: scores_per_replicate.csv and ci/power_fdr_summary_ci.csv (check 1), facts.json and,
when every check passes, consensus_signals.csv and consensus_snps.csv (one row per scenario and a
pooled row), per_replicate.csv and consensus_tables.md.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCENARIOS = [f"h2_{h}_q{q}" for h in ("030", "050", "080") for q in ("005", "015", "050")]
N_REPS = 100
N_MARKERS = 43_974
WINDOW_KB = 500
WINDOW_BP = WINDOW_KB * 1000
MLM, FARMCPU = "platform_loco", "farmcpu_2x2_E_F_nogate2_ols"   # result folders, and tool names
NAME = {MLM: "mlm", FARMCPU: "farmcpu"}
GROUPS = ["mlm_agreeing", "mlm_not_agreeing", "farmcpu_agreeing", "farmcpu_not_agreeing"]
CLASSES = ["both", "mlm_only", "farmcpu_only"]
SIGNAL_SETS = ["agreeing", "mlm", "farmcpu", "either"]
SNP_SETS = ["both", "mlm", "farmcpu", "either"]
CHECK2 = ["mlm_one_signal_each", "farmcpu_one_signal_each", "mlm_groups_add_up", "farmcpu_groups_add_up",
          "mlm_classes_add_up", "farmcpu_classes_add_up"]
GUARDS = ["truth_matches_folder", "mlm_ids_distinct", "farmcpu_ids_distinct", "mlm_positions",
          "farmcpu_positions", "mlm_rows_are_significant", "farmcpu_rows_are_significant",
          "both_same_from_each_side", "signal_true_add_up", "snp_true_add_up"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cat(*frames):
    keep = [f for f in frames if len(f)]
    return pd.concat(keep, ignore_index=True) if keep else frames[0].iloc[:0]


def tally(ev, rows, truth, snp_map, thr):
    """classify_detections on these significant rows: (number, number true, QTNs within 500 kb)."""
    c = ev.classify_detections(rows, truth, snp_map, thr, WINDOW_KB)
    return c["n_significant"], c["n_significant"] - c["n_false_positives"], c["n_true_positives"]


def clump(sig):
    """One model's signals: its significant SNPs sorted by the rule, each with the number of its
    signal, and the lead rows in the order the signals were made."""
    s = sig.sort_values(["PValue", "Chr", "Pos", "SNP"]).reset_index(drop=True)
    chrs = s["Chr"].to_numpy()
    pos = s["Pos"].to_numpy(dtype=np.int64)
    signal = np.full(len(s), -1, dtype=np.int64)
    leads = []
    for i in range(len(s)):
        if signal[i] >= 0:
            continue
        joins = (signal < 0) & (chrs == chrs[i]) & (np.abs(pos - pos[i]) <= WINDOW_BP)
        signal[joins] = len(leads)
        leads.append(i)
    return s.assign(signal=signal), s.iloc[leads].reset_index(drop=True)


def near(a, b):
    """For each lead of a: whether a lead of b is within 500 kb."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros(len(a), dtype=bool)
    same_chr = a["Chr"].to_numpy()[:, None] == b["Chr"].to_numpy()[None, :]
    dist = np.abs(a["Pos"].to_numpy(np.int64)[:, None] - b["Pos"].to_numpy(np.int64)[None, :])
    return (same_chr & (dist <= WINDOW_BP)).any(axis=1)


def replicate(ev, rd, snp_map, chr_of, pos_of, thr):
    """One replicate: its counts (rec), the two models' scores and the checks."""
    truth = json.loads((rd / "truth.json").read_text())
    data = {MLM: pd.read_csv(rd / MLM / "results.csv").dropna(subset=["PValue"]),   # evaluate_all_scenarios
            FARMCPU: pd.read_csv(rd / FARMCPU / "results.csv")}                      # as saved
    rec = dict(scenario=truth["scenario"], rep=int(truth["rep"]), h2=float(truth["h2_target"]),
               n_qtns=int(truth["n_qtns"]))
    chk = {"truth_matches_folder": rd.name == f"rep_{rec['rep']:03d}" and rd.parent.name == rec["scenario"]}
    cls, sig, members, leads, scores = {}, {}, {}, {}, []
    for m, df in data.items():
        cls[m] = ev.classify_detections(df, truth, snp_map, thr, WINDOW_KB)
        sig[m] = df[df["PValue"] < thr]
        ids = sig[m]["SNP"].astype(str)
        chk[f"{NAME[m]}_ids_distinct"] = bool(ids.is_unique)
        chk[f"{NAME[m]}_positions"] = bool((ids.map(chr_of).to_numpy() == sig[m]["Chr"].to_numpy()).all()
                                           and (ids.map(pos_of).to_numpy() == sig[m]["Pos"].to_numpy()).all())
        chk[f"{NAME[m]}_rows_are_significant"] = len(sig[m]) == cls[m]["n_significant"]
        members[m], leads[m] = clump(sig[m])
        chk[f"{NAME[m]}_one_signal_each"] = bool(
            len(members[m]) == cls[m]["n_significant"] and members[m]["SNP"].astype(str).is_unique
            and (members[m]["signal"] >= 0).all()
            and np.array_equal(np.unique(members[m]["signal"]), np.arange(len(leads[m]))))
        scores.append(dict(scenario=rec["scenario"], rep=rec["rep"], h2=rec["h2"], n_qtns=rec["n_qtns"], tool=m,
                           power=cls[m]["power"], fdr=cls[m]["fdr"], n_significant=cls[m]["n_significant"]))

    def count(rows):
        return tally(ev, rows, truth, snp_map, thr)

    # signals
    lm, lf = leads[MLM], leads[FARMCPU]
    am, af = near(lm, lf), near(lf, lm)
    n_lm, t_lm, k_lm = count(lm)
    n_lf, t_lf, k_lf = count(lf)
    rec.update(mlm_signals=n_lm, farmcpu_signals=n_lf)
    for g, rows in (("mlm_agreeing", lm[am]), ("mlm_not_agreeing", lm[~am]),
                    ("farmcpu_agreeing", lf[af]), ("farmcpu_not_agreeing", lf[~af])):
        rec[f"{g}_n"], rec[f"{g}_true"], _ = count(rows)
    chk["mlm_groups_add_up"] = rec["mlm_agreeing_n"] + rec["mlm_not_agreeing_n"] == n_lm == len(lm)
    chk["farmcpu_groups_add_up"] = rec["farmcpu_agreeing_n"] + rec["farmcpu_not_agreeing_n"] == n_lf == len(lf)
    chk["signal_true_add_up"] = (rec["mlm_agreeing_true"] + rec["mlm_not_agreeing_true"] == t_lm
                                 and rec["farmcpu_agreeing_true"] + rec["farmcpu_not_agreeing_true"] == t_lf)
    rec["signals_agreeing_qtns"] = count(cat(lm[am], lf[af]))[2]
    rec["signals_mlm_qtns"], rec["signals_farmcpu_qtns"] = k_lm, k_lf
    rec["signals_either_qtns"] = count(cat(lm, lf))[2]

    # exact SNPs
    sm, sf = sig[MLM], sig[FARMCPU]
    in_f = sm["SNP"].astype(str).isin(set(sf["SNP"].astype(str))).to_numpy()
    in_m = sf["SNP"].astype(str).isin(set(sm["SNP"].astype(str))).to_numpy()
    chk["both_same_from_each_side"] = int(in_f.sum()) == int(in_m.sum())
    classes = {"both": sm[in_f], "mlm_only": sm[~in_f], "farmcpu_only": sf[~in_m]}
    for c, rows in classes.items():
        rec[f"{c}_n"], rec[f"{c}_true"], _ = count(rows)
    n_m, n_f = cls[MLM]["n_significant"], cls[FARMCPU]["n_significant"]
    t_m, t_f = n_m - cls[MLM]["n_false_positives"], n_f - cls[FARMCPU]["n_false_positives"]
    rec.update(mlm_snps=n_m, farmcpu_snps=n_f)
    chk["mlm_classes_add_up"] = rec["both_n"] + rec["mlm_only_n"] == n_m
    chk["farmcpu_classes_add_up"] = rec["both_n"] + rec["farmcpu_only_n"] == n_f
    chk["snp_true_add_up"] = (rec["both_true"] + rec["mlm_only_true"] == t_m
                              and rec["both_true"] + rec["farmcpu_only_true"] == t_f)
    rec["snps_both_qtns"] = count(classes["both"])[2]
    rec["snps_mlm_qtns"], rec["snps_farmcpu_qtns"] = cls[MLM]["n_true_positives"], cls[FARMCPU]["n_true_positives"]
    rec["snps_either_qtns"] = count(cat(sm, classes["farmcpu_only"]))[2]
    for kind, sets in (("signals", SIGNAL_SETS), ("snps", SNP_SETS)):
        for s in sets:
            rec[f"recall_{kind}_{s}"] = rec[f"{kind}_{s}_qtns"] / rec["n_qtns"]
    return rec, scores, chk


CI_COLS = ["n_reps", "power_mean", "power_se", "power_ci_lo", "power_ci_hi",
           "fdr_mean", "fdr_se", "fdr_ci_lo", "fdr_ci_hi"]


def tie_to_committed_ci(ci_path, committed_path):
    """summarize_with_ci's TRACE-LOCO rows against the committed ones (Table S2), one per scenario:
    n_reps, and power and FDR with their SE and CI, as written (rounded to 4 decimals)."""
    keys = ["h2", "n_qtns"]
    mine = pd.read_csv(ci_path)
    saved = pd.read_csv(committed_path)
    j = mine[mine["tool"] == MLM][keys + CI_COLS].merge(
        saved[saved["tool"] == MLM][keys + CI_COLS], on=keys, how="outer",
        suffixes=("", "_committed"), indicator=True)
    rows = []
    for _, r in j.iterrows():
        both = r["_merge"] == "both"
        rows.append(dict(h2=float(r["h2"]), qtns=int(r["n_qtns"]), matched=bool(both),
                         equal=bool(both and all(r[c] == r[c + "_committed"] for c in CI_COLS))))
    return rows


def tie_to_saved(scores, sources):
    """Per replicate, MLM's power, FDR and number significant against the committed scores: the
    count exactly, power and FDR to 1e-15 (the saved digits); the exactly equal rows are counted too."""
    cols = ["scenario", "rep", "power", "fdr", "n_significant"]
    b = pd.read_csv(sources["bonferroni_results"], float_precision="round_trip")
    saved = {MLM: b[b["tool"] == MLM][cols]}
    out = {}
    for m in (MLM,):
        mine = scores[scores["tool"] == m][cols]
        j = mine.merge(saved[m], on=["scenario", "rep"], how="outer", suffixes=("", "_saved"), indicator=True)
        both = (j["_merge"] == "both").to_numpy()
        same_n = (j["n_significant"] == j["n_significant_saved"]).to_numpy()
        exact = both & same_n & (j["power"] == j["power_saved"]).to_numpy() & (j["fdr"] == j["fdr_saved"]).to_numpy()
        equal = both & same_n & np.isclose(j["power"], j["power_saved"], rtol=0, atol=1e-15) \
            & np.isclose(j["fdr"], j["fdr_saved"], rtol=0, atol=1e-15)
        out[NAME[m]] = dict(replicates=int(len(mine)), saved_rows=int(len(saved[m])), matched=int(both.sum()),
                            equal=int(equal.sum()), exactly_equal=int(exact.sum()),
                            max_abs_diff_power=float((j["power"] - j["power_saved"]).abs().max()),
                            max_abs_diff_fdr=float((j["fdr"] - j["fdr_saved"]).abs().max()))
    return out


def summary(pr, units, totals, sets, kind):
    """One row per scenario and a pooled row: per group or class the total, the median per
    replicate, the number true and the precision; the recall of each set."""
    rows = []
    for label, sub in [(sc, pr[pr["scenario"] == sc]) for sc in SCENARIOS] + [("all_9", pr)]:
        pooled = label == "all_9"
        row = dict(scenario=label, h2=None if pooled else float(sub["h2"].iloc[0]),
                   n_qtns=None if pooled else int(sub["n_qtns"].iloc[0]), n_reps=len(sub))
        for t in totals:
            row[t] = int(sub[t].sum())
        for u in units:
            n, t = int(sub[f"{u}_n"].sum()), int(sub[f"{u}_true"].sum())
            row.update({f"{u}_n": n, f"{u}_median": float(sub[f"{u}_n"].median()), f"{u}_true": t,
                        f"{u}_precision": t / n if n else np.nan})
        for s in sets:
            row[f"recall_{s}"] = float(sub[f"recall_{kind}_{s}"].mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df["n_qtns"] = df["n_qtns"].astype("Int64")
    return df


def pct(fr):
    """A fraction as a percentage to one decimal, rounded half up from the exact value."""
    d = (Decimal(fr.numerator) * 100 / Decimal(fr.denominator)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{d}%"


def num(x):
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.1f}"


def table(pr, first, units, labels, count_label, kind, sets, set_labels):
    """The report's table: the measures as rows, the 9 scenarios and the 900 replicates as columns."""
    cols = [(f"{int(sc[3:6]) / 100:.1f}/{int(sc[8:])}", pr[pr["scenario"] == sc]) for sc in SCENARIOS]
    cols.append(("All 9", pr))
    lines = [f"| {first} | Measure | " + " | ".join(c for c, _ in cols) + " |",
             "| --- | --- | " + " | ".join("---:" for _ in cols) + " |"]
    for u, lab in zip(units, labels):
        n = [int(sub[f"{u}_n"].sum()) for _, sub in cols]
        t = [int(sub[f"{u}_true"].sum()) for _, sub in cols]
        lines.append(f"| {lab} | {count_label} | " + " | ".join(f"{v:,}" for v in n) + " |")
        lines.append("| | median per replicate | " + " | ".join(num(sub[f"{u}_n"].median()) for _, sub in cols) + " |")
        lines.append("| | true | " + " | ".join(f"{v:,}" for v in t) + " |")
        lines.append("| | precision | " + " | ".join(pct(Fraction(b, a)) if a else "–" for a, b in zip(n, t)) + " |")
    for i, (s, lab) in enumerate(zip(sets, set_labels)):
        vals = []
        for _, sub in cols:
            share = sum((Fraction(int(k), int(q)) for k, q in zip(sub[f"{kind}_{s}_qtns"], sub["n_qtns"])), Fraction(0))
            vals.append(pct(share / len(sub)))
        lines.append(f"| {'Recall' if i == 0 else ''} | {lab} | " + " | ".join(vals) + " |")
    return "\n".join(lines)


def tables_md(pr):
    signals = table(pr, "Group", GROUPS,
                    ["MLM signals with an agreeing FarmCPU signal", "MLM signals without one",
                     "FarmCPU signals with an agreeing MLM signal", "FarmCPU signals without one"],
                    "signals", "signals", SIGNAL_SETS,
                    ["agreeing signals", "all MLM signals", "all FarmCPU signals", "either model"])
    snps = table(pr, "Class", CLASSES, ["Both", "MLM only", "FarmCPU only"], "SNPs", "snps", SNP_SETS,
                 ["both", "all MLM SNPs", "all FarmCPU SNPs", "either model"])
    note = ("Columns: h²/QTNs, 100 replicates each; All 9: the 900 replicates pooled. Precision and recall "
            "are rounded half up from the exact counts; – : no signal or SNP in the group.")
    return (f"**Signals (main analysis).** {note}\n\n{signals}\n\n"
            f"**Exact SNPs (secondary analysis).** The same columns.\n\n{snps}\n")


def main():
    t0 = time.perf_counter()
    started = datetime.now().isoformat(timespec="seconds")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the work folder")
    ap.add_argument("--sim-data", default=str(HERE / "sim_data"))
    args = ap.parse_args()
    out, sim = Path(args.out), Path(args.sim_data).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # the scoring module that made Tables S2 and S12, imported unchanged, writing no bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    import evaluate_simulation as ev   # noqa: E402
    if Path(ev.__file__).resolve() != (HERE / "evaluate_simulation.py").resolve():
        raise SystemExit(f"imported the wrong evaluate_simulation: {ev.__file__}")
    sources = {"bonferroni_results": HERE / "sim_summary" / "bonferroni_results.csv",
               "power_fdr_summary_ci": HERE / "sim_summary" / "power_fdr_summary_ci.csv"}

    snp_map = pd.read_csv(sim / "snp_map.csv")
    thr = 0.05 / len(snp_map)
    ids = snp_map["SNP_ID"].astype(str)
    chr_of = pd.Series(snp_map["Chr"].to_numpy(), index=ids)
    pos_of = pd.Series(snp_map["Pos"].to_numpy(), index=ids)
    facts = dict(started=started, script=str(Path(__file__).resolve()), script_sha256=sha256(__file__),
                 python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                 scoring_module=str(Path(ev.__file__).resolve()), scoring_module_sha256=sha256(ev.__file__),
                 sim_data=str(sim), snp_map_sha256=sha256(sim / "snp_map.csv"), n_markers=len(snp_map),
                 threshold=thr, window_kb=WINDOW_KB,
                 score_sources={k: dict(path=str(p), sha256=sha256(p)) for k, p in sources.items()})
    guards = {"n_markers": len(snp_map) == N_MARKERS, "snp_map_ids_distinct": bool(ids.is_unique)}

    recs, scores, passed, failed = [], [], {}, {}
    for sc in SCENARIOS:
        for k in range(N_REPS):
            rd = sim / sc / f"rep_{k:03d}"
            rec, sco, chk = replicate(ev, rd, snp_map, chr_of, pos_of, thr)
            recs.append(rec)
            scores.extend(sco)
            for name, ok in chk.items():
                passed[name] = passed.get(name, 0) + int(bool(ok))
                if not ok:
                    failed.setdefault(name, []).append(f"{sc}/rep_{k:03d}")
    n_rep = len(recs)
    pr = pd.DataFrame(recs)
    sco = pd.DataFrame(scores)

    # check 1 (MLM): the committed scores per replicate, then the committed rows through summarize_with_ci
    sco.to_csv(out / "scores_per_replicate.csv", index=False, lineterminator="\n")
    (out / "ci").mkdir(exist_ok=True)
    ev.summarize_with_ci(results_path=str(out / "scores_per_replicate.csv"), output_dir=str(out / "ci"))
    rows = tie_to_committed_ci(out / "ci" / "power_fdr_summary_ci.csv", sources["power_fdr_summary_ci"])
    tie = tie_to_saved(sco, sources)
    check1 = (len(rows) == len(SCENARIOS) and all(r["matched"] and r["equal"] for r in rows)
              and all(t["replicates"] == t["saved_rows"] == t["matched"] == t["equal"] == n_rep for t in tie.values()))
    check2 = all(passed.get(n, 0) == n_rep for n in CHECK2)
    guards_ok = all(guards.values()) and all(passed.get(n, 0) == n_rep for n in GUARDS)
    ok = n_rep == len(SCENARIOS) * N_REPS and check1 and check2 and guards_ok

    facts.update(replicates=n_rep, check1=dict(passed=check1, ci_rows=rows, per_replicate=tie),
                 check2=dict(passed=check2, replicates_passing={n: passed.get(n, 0) for n in CHECK2}),
                 guards=dict(passed=guards_ok, **guards, replicates_passing={n: passed.get(n, 0) for n in GUARDS}),
                 failed_replicates={n: v[:20] for n, v in failed.items()},
                 totals={c: int(pr[c].sum()) for c in ("mlm_snps", "farmcpu_snps", "mlm_signals", "farmcpu_signals")})
    if ok:
        summary(pr, GROUPS, ["mlm_signals", "farmcpu_signals"], SIGNAL_SETS, "signals").to_csv(
            out / "consensus_signals.csv", index=False, lineterminator="\n")
        summary(pr, CLASSES, ["mlm_snps", "farmcpu_snps"], SNP_SETS, "snps").to_csv(
            out / "consensus_snps.csv", index=False, lineterminator="\n")
        pr.to_csv(out / "per_replicate.csv", index=False, lineterminator="\n")
        with open(out / "consensus_tables.md", "w", encoding="utf-8", newline="\n") as fh:
            fh.write(tables_md(pr))
    facts.update(status="pass" if ok else "fail", finished=datetime.now().isoformat(timespec="seconds"),
                 elapsed_s=round(time.perf_counter() - t0, 1))
    with open(out / "facts.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(facts, fh, indent=1, ensure_ascii=False)
    print(f"\n{facts['status']}: {n_rep} replicates; check 1 {check1}, check 2 {check2}, guards {guards_ok}; "
          f"{facts['elapsed_s']} s; {out}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
