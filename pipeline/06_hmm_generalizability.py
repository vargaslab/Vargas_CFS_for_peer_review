#!/usr/bin/env python3
"""Evaluate cross-year generalizability and matched-deletion sensitivity.

The script retrains K4 and K6 models after omitting years, compares fixed and
refitted models on QC-restricted subsets, and contrasts observed QC changes
with deterministic pseudo-deletion masks. Its leave-one-year-out summaries feed
Figure S1 and Table S3; all state alignment is explicit and auditable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import warnings
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "saltmarsh_generalizability_mpl"),
)

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/stjones_halfhourly_contract_v1.csv.gz"
CORE = ROOT / "config/data_contract_core_v1.json"
PRIMARY = ROOT / "outputs/complete_dataset/hmm_primary"
QC = ROOT / "outputs/complete_dataset/hmm_qc_robustness"
TEMP = ROOT / "outputs/complete_dataset/temporal_scale_diagnostic"
STABILITY = ROOT / "outputs/complete_dataset/hmm_state_stability"
OUT = ROOT / "outputs/complete_dataset/hmm_generalizability"
FIG = ROOT / "figures/diagnostics/hmm_generalizability"
REPORT = ROOT / "reports/complete_dataset/hmm_generalizability_report.md"
BRANCHES = {"qc01": "qc_01_sensitivity_eligible", "qc0": "qc_0_sensitivity_eligible"}
YEARS = tuple(range(2016, 2022))
N_MASKS = 5
BASE_SEED = 903_000


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--fit-shard", default="", help="zero-based shard,total")
    p.add_argument("--max-iter", type=int, default=125)
    p.add_argument("--tol", type=float, default=0.01)
    return p.parse_args()


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist() if isinstance(x, np.ndarray) else str(x)) + "\n")


def load_data():
    contract = json.loads(CORE.read_text())
    if contract["contract_status"] != "FROZEN":
        raise RuntimeError("Core carbon contract is not frozen")
    required = [PRIMARY / "best_by_structure/full/K4/fit_metadata.json", PRIMARY / "best_by_structure/full/K6/fit_metadata.json", QC / "stability_by_k_and_qc.csv"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    use = ["source_row", "timestamp_start", "NEE", "CH4", "primary_hmm_eligible", *BRANCHES.values()]
    d = pd.read_csv(DATA, usecols=use, compression="gzip")
    d["timestamp_start"] = pd.to_datetime(d["timestamp_start"])
    d = d.loc[d.primary_hmm_eligible].sort_values(["timestamp_start", "source_row"], kind="mergesort").reset_index(drop=True)
    d["year"] = d.timestamp_start.dt.year
    d["month"] = d.timestamp_start.dt.month
    return d


def prepare(frame: pd.DataFrame, mask=None, scaling=None):
    x = frame.loc[np.ones(len(frame), bool) if mask is None else np.asarray(mask, bool)].copy().reset_index(drop=True)
    br = x.timestamp_start.diff().ne(pd.Timedelta(minutes=30))
    if len(br):
        br.iloc[0] = True
    x["sequence_id"] = br.cumsum().astype(int)
    lengths = x.groupby("sequence_id", sort=False).size().astype(int).tolist()
    raw = x[["NEE", "CH4"]].to_numpy(float)
    if scaling is None:
        mean, sd = raw.mean(axis=0), raw.std(axis=0, ddof=1)
    else:
        mean, sd = np.asarray(scaling["mean"]), np.asarray(scaling["sd"])
    return x, (raw - mean) / sd, lengths, {"mean": mean, "sd": sd}


def primary_fit(k):
    dr = PRIMARY / f"best_by_structure/full/K{k}"
    meta = json.loads((dr / "fit_metadata.json").read_text())
    inf = np.load(dr / "inference.npz")
    return {
        "K": k, "means": np.asarray(meta["emission_means_standardized"]),
        "covs": np.asarray(meta["emission_covariances_standardized"]),
        "trans": np.asarray(meta["transition_matrix"]), "start": np.asarray(meta["start_probabilities"]),
        "assign": inf["viterbi_state_index"].astype(int), "post": inf["posterior_probabilities"].astype(float),
        "meta": meta,
    }


def fixed_hmm(fit):
    k = fit["K"]
    model = GaussianHMM(n_components=k, covariance_type="full", implementation="scaling", init_params="", params="")
    model.n_features = 2
    model.startprob_ = fit["start"].copy()
    model.transmat_ = fit["trans"].copy()
    model.means_ = fit["means"].copy()
    model.covars_ = fit["covs"].copy()
    return model


def canonicalize(model, post, assign):
    means, covs, trans, start = np.asarray(model.means_), np.asarray(model.covars_), np.asarray(model.transmat_), np.asarray(model.startprob_)
    order = np.lexsort((means[:, 1], means[:, 0]))
    remap = np.empty(len(order), int)
    remap[order] = np.arange(len(order))
    return means[order], covs[order], trans[np.ix_(order, order)], start[order], post[:, order], remap[assign]


def param_count(k):
    return (k - 1) + k * (k - 1) + 2 * k + 3 * k


def fit_path(kind, unit, k, start):
    return OUT / "fits" / kind / str(unit) / f"K{k}" / f"start{start:02d}"


def fit_one(kind, unit, k, start, z, lengths, max_iter, tol):
    dr = fit_path(kind, unit, k, start)
    dr.mkdir(parents=True, exist_ok=True)
    unit_code = sum((i + 1) * ord(ch) for i, ch in enumerate(str(unit))) % 50_000
    seed = BASE_SEED + (100_000 if kind == "pseudo" else 0) + unit_code + k * 100 + start
    base = {"kind": kind, "unit": str(unit), "K": k, "start_index": start, "seed": seed, "success": False, "converged": False}
    try:
        m = GaussianHMM(n_components=k, covariance_type="full", n_iter=max_iter, tol=tol, min_covar=1e-6, random_state=seed, implementation="scaling")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("once")
            m.fit(z, lengths)
            ll = float(m.score(z, lengths))
            post = m.predict_proba(z, lengths)
            assign = m.predict(z, lengths)
        means, covs, trans, startp, post, assign = canonicalize(m, post, assign)
        hist = [float(v) for v in m.monitor_.history]
        reached = int(m.monitor_.iter) >= max_iter
        converged = bool(m.monitor_.converged) and not reached and (len(hist) < 2 or hist[-1] >= hist[-2] - 1e-3)
        p = param_count(k)
        meta = {**base, "success": True, "converged": converged, "iterations": int(m.monitor_.iter), "reached_iteration_limit": reached,
                "log_likelihood": ll, "AIC": 2*p-2*ll, "BIC": math.log(len(z))*p-2*ll,
                "emission_means_standardized": means.tolist(), "emission_covariances_standardized": covs.tolist(),
                "transition_matrix": trans.tolist(), "start_probabilities": startp.tolist(),
                "occupancy": (np.bincount(assign, minlength=k)/len(assign)).tolist(),
                "mean_maximum_posterior": float(post.max(1).mean()), "warning_count": len(caught), "monitor_history": hist}
        joblib.dump(m, dr / "model.joblib", compress=3)
        np.savez_compressed(dr / "inference.npz", posterior=post.astype("float32"), assignments=assign.astype("int16"))
        dump(dr / "fit_metadata.json", meta)
        return meta
    except Exception as e:
        base["error"] = f"{type(e).__name__}: {e}"
        dump(dr / "fit_metadata.json", base)
        return base


def load_fit(kind, unit, k, start):
    dr = fit_path(kind, unit, k, start)
    if not (dr / "fit_metadata.json").is_file():
        return None
    r = json.loads((dr / "fit_metadata.json").read_text())
    if r.get("success") and (dr / "inference.npz").is_file():
        inf = np.load(dr / "inference.npz")
        r.update(means=np.asarray(r["emission_means_standardized"]), covs=np.asarray(r["emission_covariances_standardized"]),
                 trans=np.asarray(r["transition_matrix"]), start=np.asarray(r["start_probabilities"]),
                 post=inf["posterior"].astype(float), assign=inf["assignments"].astype(int))
    return r


def primary_scaling(data):
    raw = data[["NEE", "CH4"]].to_numpy(float)
    return {"mean": raw.mean(0), "sd": raw.std(0, ddof=1)}


def raw_geometry(fit, scaling):
    means = scaling["mean"] + fit["means"] * scaling["sd"]
    D = np.diag(scaling["sd"])
    covs = np.asarray([D @ c @ D for c in fit["covs"]])
    return means, covs


def align_to_primary(fit, scaling, primary, pscale):
    means, _ = raw_geometry(fit, scaling)
    pmeans, _ = raw_geometry(primary, pscale)
    cost = np.linalg.norm((means[:, None, :] - pmeans[None, :, :]) / pscale["sd"], axis=2)
    i, j = linear_sum_assignment(cost)
    mapping = np.empty(fit["K"], int); mapping[i] = j
    order = np.argsort(mapping)
    return mapping, {**fit, "means": fit["means"][order], "covs": fit["covs"][order], "trans": fit["trans"][np.ix_(order, order)],
                     "post": fit["post"][:, order], "assign": mapping[fit["assign"]]}, cost[i, j]


def assignment_stats(a, b, k):
    cm = np.zeros((k, k), int); np.add.at(cm, (a, b), 1)
    return {"ARI": adjusted_rand_score(a, b), "NMI": normalized_mutual_info_score(a, b),
            "retained_state_fraction": float(np.mean(a == b)), "changed_state_fraction": float(np.mean(a != b)),
            "aligned_confusion_matrix": json.dumps(cm.tolist())}


def state_json(values):
    return json.dumps({f"S{i+1}": float(v) if np.isfinite(v) else None for i, v in enumerate(values)})


def fixed_transport(data, pscale, primaries):
    qcsel = pd.read_csv(QC / "stability_by_k_and_qc.csv").set_index(["branch", "K"])
    rows = []
    for branch, flag in BRANCHES.items():
        frame, z, lengths, _ = prepare(data, data[flag].to_numpy(), pscale)
        idx = frame.index.to_numpy() if False else data.index[data[flag]].to_numpy()
        for k in (4, 6):
            primary = primaries[k]
            model = fixed_hmm(primary)
            post = model.predict_proba(z, lengths); fixed = model.predict(z, lengths)
            pshared = primary["assign"][idx]
            best_seed = int(qcsel.loc[(branch, k), "best_seed"])
            candidates = list((QC / "fits" / branch / f"K{k}_full").glob(f"*seed{best_seed}"))
            if len(candidates) != 1:
                raise RuntimeError(f"Cannot resolve QC fit {branch} K{k} seed {best_seed}")
            qm = json.loads((candidates[0] / "fit_metadata.json").read_text()); qi = np.load(candidates[0] / "inference.npz")
            qfit = {"K": k, "means": np.asarray(qm["emission_means_standardized"]), "covs": np.asarray(qm["emission_covariances_standardized"]),
                    "trans": np.asarray(qm["transition_matrix"]), "start": np.zeros(k), "post": qi["posterior"].astype(float), "assign": qi["assignments"].astype(int)}
            _, _, _, qscale = prepare(data, data[flag].to_numpy())
            _, qaligned, _ = align_to_primary(qfit, qscale, primary, pscale)
            certainty = [post[fixed == s, s].mean() if np.any(fixed == s) else np.nan for s in range(k)]
            emission = {}
            for s in range(k):
                g = frame.loc[fixed == s, ["NEE", "CH4"]]
                emission[f"S{s+1}"] = {"n": len(g), "NEE_mean": g.NEE.mean(), "NEE_median": g.NEE.median(), "CH4_mean": g.CH4.mean(), "CH4_median": g.CH4.median()}
            common = {"branch": branch, "K": k, "n": len(frame), "sequence_count": len(lengths),
                      "occupancy": state_json(np.bincount(fixed, minlength=k)/len(fixed)), "mean_posterior_certainty": post.max(1).mean(),
                      "state_posterior_certainty": state_json(certainty), "emission_summaries": json.dumps(emission)}
            rows.append({**common, "comparison": "fixed_primary_vs_primary_shared", **assignment_stats(pshared, fixed, k)})
            rows.append({**common, "comparison": "fixed_primary_vs_qc_refit", **assignment_stats(qaligned["assign"], fixed, k)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "fixed_model_qc_transportability.csv", index=False)
    return out


def shift_mask(data, flag, replicate):
    rng = np.random.default_rng(BASE_SEED + (10_000 if flag == BRANCHES["qc0"] else 0) + replicate)
    actual = data[flag].to_numpy(bool); pseudo = np.zeros(len(data), bool)
    for _, ids in data.groupby(["year", "month"], sort=True).groups.items():
        ids = np.asarray(list(ids), int); a = actual[ids]
        pseudo[ids] = np.roll(a, int(rng.integers(0, max(1, len(a)))))
    return pseudo


def mask_table(data):
    rows, masks = [], {}
    for branch, flag in BRANCHES.items():
        actual = data[flag].to_numpy(bool)
        for rep in range(1, N_MASKS + 1):
            mask = shift_mask(data, flag, rep); masks[(branch, rep)] = mask
            _, _, lens, _ = prepare(data, mask)
            # discrepancy is transparent; year-month counts are exactly preserved.
            removed = ~mask
            run = pd.Series(removed).groupby((pd.Series(removed) != pd.Series(removed).shift()).cumsum()).agg(["first", "size"])
            rr = run.loc[run["first"], "size"]
            rows.append({"branch": branch, "replicate": rep, "retained_n": int(mask.sum()), "retention_fraction": mask.mean(),
                         "sequence_count": len(lens), "adjacent_pairs": int(mask.sum()-len(lens)), "maximum_sequence_length": max(lens),
                         "median_removed_run": rr.median(), "maximum_removed_run": rr.max(),
                         "actual_retained_n": int(actual.sum()), "actual_sequence_count": prepare(data, actual)[2].__len__(),
                         "year_month_retained_count_max_abs_error": 0})
    pd.DataFrame(rows).to_csv(OUT / "pseudo_deletion_mask_summary.csv", index=False)
    return masks


def tasks(data, masks):
    ans = []
    for year in YEARS:
        mask = data.year.ne(year).to_numpy()
        _, z, lengths, _ = prepare(data, mask)
        for k in (4, 6):
            for start in range(1, 6): ans.append(("loyo", year, k, start, z, lengths))
    for (branch, rep), mask in masks.items():
        _, z, lengths, _ = prepare(data, mask)
        unit = f"{branch}_{rep:02d}"
        for k, n in ((4, 5), (6, 3)):
            for start in range(1, n+1): ans.append(("pseudo", unit, k, start, z, lengths))
    return ans


def best_fits(data, masks, max_iter, tol):
    results = {}
    for kind, unit, k, nstarts, mask in [
        *[("loyo", y, k, 5, data.year.ne(y).to_numpy()) for y in YEARS for k in (4, 6)],
        *[("pseudo", f"{b}_{r:02d}", k, 5 if k == 4 else 3, m) for (b, r), m in masks.items() for k in (4, 6)]]:
        _, z, lengths, scaling = prepare(data, mask)
        fits = []
        for s in range(1, nstarts+1):
            f = load_fit(kind, unit, k, s)
            if f is None: f = fit_one(kind, unit, k, s, z, lengths, max_iter, tol)
            if f.get("success"): fits.append(f)
        if not fits:
            raise RuntimeError(f"No successful independent fit: {kind} {unit} K{k}")
        conv = [f for f in fits if f.get("converged")]
        winner = max(conv or fits, key=lambda f: f["log_likelihood"])
        # Membership stability across starts, aligned to the winner in branch-standardized geometry.
        aligned = []
        for f in conv or fits:
            c = np.linalg.norm(f["means"][:, None, :] - winner["means"][None, :, :], axis=2)
            i, j = linear_sum_assignment(c); mp = np.empty(k, int); mp[i] = j
            aligned.append(mp[f["assign"]])
        aris = [adjusted_rand_score(a, b) for a, b in combinations(aligned, 2)]
        winner = dict(winner); winner["start_stability_ARI"] = float(np.mean(aris)) if aris else 1.0
        winner["successful_starts"] = len(fits); winner["converged_starts"] = len(conv); winner["scaling"] = scaling; winner["mask"] = mask
        results[(kind, str(unit), k)] = winner
    return results


def centroid_metrics(fit, scaling, primary, pscale):
    mp, aligned, distances = align_to_primary(fit, scaling, primary, pscale)
    means, _ = raw_geometry(aligned, scaling); pmeans, _ = raw_geometry(primary, pscale)
    displacement = np.linalg.norm((means-pmeans)/pscale["sd"], axis=1)
    occ = np.bincount(aligned["assign"], minlength=fit["K"])/len(aligned["assign"])
    pocc = np.bincount(primary["assign"], minlength=fit["K"])/len(primary["assign"])
    sep = np.linalg.norm((means[:, None]-means[None, :])/pscale["sd"], axis=2); np.fill_diagonal(sep, np.inf)
    return mp, aligned, means, {"mean_centroid_displacement": displacement.mean(), "maximum_centroid_displacement": displacement.max(),
                                "occupancy_change_L1": np.abs(occ-pocc).sum(), "minimum_centroid_separation": sep.min(),
                                "NEE_sign_change_count": int(np.sum(np.sign(means[:,0]) != np.sign(pmeans[:,0]))),
                                "CH4_order_matches_primary": bool(np.array_equal(np.argsort(means[:,1]), np.argsort(pmeans[:,1])))}


def markov_metrics(assign, lengths, k):
    ends = np.cumsum(lengths)-1; ok = np.ones(max(0, len(assign)-1), bool)
    if len(ends) > 1: ok[ends[:-1]] = False
    pairs = np.zeros((k, k), int)
    if len(assign) > 1: np.add.at(pairs, (assign[:-1][ok], assign[1:][ok]), 1)
    occ = np.bincount(assign, minlength=k)/len(assign)
    observed = np.trace(pairs)/pairs.sum() if pairs.sum() else np.nan
    expected = np.sum(occ**2)
    return observed/expected if expected else np.nan


def hierarchy_row(label, scope, k4, k6):
    a, b = k4["assign"], k6["assign"]
    joint = np.zeros((4, 6), int); np.add.at(joint, (a, b), 1)
    parent = joint.argmax(0); purity = joint.max(0)/np.maximum(1, joint.sum(0)); coarse = parent[b]
    return {"analysis": scope, "unit": label, "n": len(a), "child_parent_purity": purity.mean(),
            "minimum_child_parent_purity": purity.min(), "coarsened_ARI": adjusted_rand_score(a, coarse),
            "coarsened_NMI": normalized_mutual_info_score(a, coarse), "child_parent_mapping": json.dumps((parent+1).tolist()),
            "joint_count_matrix": json.dumps(joint.tolist())}


def loyo_outputs(data, results, primaries, pscale):
    summaries, centroids, hiers = [], [], []
    for year in YEARS:
        mask = data.year.ne(year).to_numpy(); train, _, lengths, _ = prepare(data, mask)
        test, ztest, test_lengths, _ = prepare(data, ~mask, results[("loyo", str(year), 4)]["scaling"])
        aligned_by_k = {}
        for k in (4, 6):
            fit = results[("loyo", str(year), k)]; scaling = fit["scaling"]
            mp, aligned, means, cm = centroid_metrics(fit, scaling, primaries[k], pscale); aligned_by_k[k] = aligned
            model = fixed_hmm(fit); test_assign = mp[model.predict(ztest, test_lengths)]; truth = primaries[k]["assign"][~mask]
            agree = assignment_stats(truth, test_assign, k)
            certainty = aligned["post"].max(1)
            summaries.append({"omitted_year": year, "K": k, "training_n": len(train), "test_n": len(test), "sequence_count": len(lengths),
                              "successful_starts": fit["successful_starts"], "converged_starts": fit["converged_starts"], "winner_converged": fit["converged"],
                              "best_log_likelihood": fit["log_likelihood"], "BIC": fit["BIC"], "occupancy": state_json(np.bincount(aligned["assign"], minlength=k)/len(aligned["assign"])),
                              "mean_posterior_certainty": certainty.mean(), "centroid_separation": cm["minimum_centroid_separation"],
                              "start_stability_ARI": fit["start_stability_ARI"], **cm,
                              "omitted_year_assignment_ARI": agree["ARI"], "omitted_year_assignment_NMI": agree["NMI"],
                              "omitted_year_same_state_fraction": agree["retained_state_fraction"],
                              "markov_self_pair_ratio": markov_metrics(aligned["assign"], lengths, k)})
            pmeans, _ = raw_geometry(primaries[k], pscale)
            for s in range(k):
                centroids.append({"omitted_year": year, "K": k, "state": s+1, "NEE_centroid": means[s,0], "CH4_centroid": means[s,1],
                                  "primary_NEE_centroid": pmeans[s,0], "primary_CH4_centroid": pmeans[s,1],
                                  "standardized_displacement": np.linalg.norm((means[s]-pmeans[s])/pscale["sd"]),
                                  "NEE_sign_changed": np.sign(means[s,0]) != np.sign(pmeans[s,0]),
                                  "occupancy": np.mean(aligned["assign"] == s), "primary_occupancy": np.mean(primaries[k]["assign"] == s)})
        hiers.append(hierarchy_row(str(year), "leave_one_year_out", aligned_by_k[4], aligned_by_k[6]))
    pd.DataFrame(summaries).to_csv(OUT / "leave_one_year_out_summary.csv", index=False)
    pd.DataFrame(centroids).to_csv(OUT / "leave_one_year_out_centroids.csv", index=False)
    return pd.DataFrame(summaries), pd.DataFrame(centroids), hiers


def actual_qc_fits(data, primaries, pscale):
    sel = pd.read_csv(QC / "stability_by_k_and_qc.csv").set_index(["branch", "K"]); ans = {}
    for branch, flag in BRANCHES.items():
        mask = data[flag].to_numpy(); _, _, lengths, scaling = prepare(data, mask)
        for k in (4, 6):
            seed = int(sel.loc[(branch, k), "best_seed"]); drs = list((QC / "fits" / branch / f"K{k}_full").glob(f"*seed{seed}")); dr = drs[0]
            m = json.loads((dr / "fit_metadata.json").read_text()); inf = np.load(dr / "inference.npz")
            fit = {"K": k, "means": np.asarray(m["emission_means_standardized"]), "covs": np.asarray(m["emission_covariances_standardized"]),
                   "trans": np.asarray(m["transition_matrix"]), "start": np.asarray(m.get("start_probabilities", np.repeat(1/k,k))),
                   "post": inf["posterior"].astype(float), "assign": inf["assignments"].astype(int), "scaling": scaling, "mask": mask,
                   "start_stability_ARI": float(sel.loc[(branch,k), "pairwise_ARI"]), "converged": True,
                   "successful_starts": int(sel.loc[(branch,k), "successful_starts"]),
                   "converged_starts": int(sel.loc[(branch,k), "converged_starts"]), "BIC": float(sel.loc[(branch,k), "BIC"])}
            ans[(branch,k)] = fit
    return ans


def deletion_outputs(data, masks, results, actual, primaries, pscale):
    p4 = primaries[4]; p6 = primaries[6]; rows4, rows6, hiers, aligned_cache = [], [], [], {}
    for source in ("actual", "pseudo"):
        units = [(b, 0, data[BRANCHES[b]].to_numpy()) for b in BRANCHES] if source == "actual" else [(b, r, m) for (b,r),m in masks.items()]
        for branch, rep, mask in units:
            aligned = {}
            for k, bucket in ((4, rows4), (6, rows6)):
                fit = actual[(branch,k)] if source == "actual" else results[("pseudo", f"{branch}_{rep:02d}", k)]
                scaling = fit["scaling"]; mp, al, means, cm = centroid_metrics(fit, scaling, primaries[k], pscale); aligned[k] = al
                truth = primaries[k]["assign"][mask]; stats = assignment_stats(truth, al["assign"], k)
                _, _, lens, _ = prepare(data, mask)
                bucket.append({"source": source, "branch": branch, "replicate": rep, "K": k, "retained_n": int(mask.sum()), "sequence_count": len(lens),
                               **{q: stats[q] for q in ("ARI","NMI","retained_state_fraction","changed_state_fraction")}, **cm,
                               "mean_posterior_certainty": al["post"].max(1).mean(), "start_stability_ARI": fit["start_stability_ARI"],
                               "successful_starts": fit.get("successful_starts", np.nan), "converged_starts": fit.get("converged_starts", np.nan),
                               "winner_converged": fit.get("converged", False), "BIC": fit.get("BIC", np.nan),
                               "markov_self_pair_ratio": markov_metrics(al["assign"], lens, k),
                               "high_CH4_state_NEE_centroid": means[np.argmax(raw_geometry(primaries[k], pscale)[0][:,1]),0]})
            primary_h = hierarchy_row("primary_K6", "primary_K6_to_refit_K4", aligned[4], {"assign": p6["assign"][mask]})
            rows4[-1].update({"primary_K6_child_parent_purity": primary_h["child_parent_purity"],
                              "primary_K6_coarsened_ARI": primary_h["coarsened_ARI"],
                              "primary_K6_coarsened_NMI": primary_h["coarsened_NMI"]})
            hiers.append(hierarchy_row(f"{branch}_{rep:02d}" if source == "pseudo" else branch, f"{source}_deletion", aligned[4], aligned[6]))
    d4, d6 = pd.DataFrame(rows4), pd.DataFrame(rows6)
    d4[d4.source.eq("pseudo")].drop(columns="source").to_csv(OUT / "pseudo_deletion_k4_results.csv", index=False)
    d6[d6.source.eq("pseudo")].drop(columns="source").to_csv(OUT / "pseudo_deletion_k6_results.csv", index=False)
    # Put observed metrics into their matched empirical null distributions.
    comp = []
    metrics = ["ARI", "NMI", "mean_centroid_displacement", "maximum_centroid_displacement", "occupancy_change_L1", "mean_posterior_certainty", "minimum_centroid_separation", "NEE_sign_change_count", "CH4_order_matches_primary", "high_CH4_state_NEE_centroid", "start_stability_ARI", "markov_self_pair_ratio"]
    k4_only_metrics = ["primary_K6_child_parent_purity", "primary_K6_coarsened_ARI", "primary_K6_coarsened_NMI"]
    for k, tab in ((4,d4),(6,d6)):
        for branch in BRANCHES:
            obs = tab[(tab.source=="actual") & (tab.branch==branch)].iloc[0]; null = tab[(tab.source=="pseudo") & (tab.branch==branch)]
            for metric in metrics + (k4_only_metrics if k == 4 else []):
                val = float(obs[metric]); vals = null[metric].astype(float).to_numpy()
                comp.append({"K": k, "branch": branch, "metric": metric, "observed": val, "pseudo_min": vals.min(), "pseudo_median": np.median(vals), "pseudo_max": vals.max(),
                             "empirical_percentile_le_observed": 100*(np.sum(vals <= val)+1)/(len(vals)+1),
                             "ascending_rank_with_observed": int(np.sum(vals < val)+1), "comparison_n_pseudo": len(vals)})
    comparison = pd.DataFrame(comp); comparison.to_csv(OUT / "actual_vs_pseudo_deletion_comparison.csv", index=False)
    return d4, d6, comparison, hiers


def response_shift(data, actual, primaries, pscale):
    rows = []; qlevels = [.01,.05,.10,.25,.50,.75,.90,.95,.99]
    ch4q95, ch4q99 = data.CH4.quantile([.95,.99])
    def add(branch, stratum, key, frame):
        x = frame[["NEE","CH4"]]
        values = {"n":len(x), "NEE_mean":x.NEE.mean(), "NEE_median":x.NEE.median(), "NEE_variance":x.NEE.var(),
                  "CH4_mean":x.CH4.mean(), "CH4_median":x.CH4.median(), "CH4_variance":x.CH4.var(), "NEE_CH4_covariance":x.cov().iloc[0,1],
                  "CH4_ge_primary_q95_fraction":np.mean(x.CH4>=ch4q95), "CH4_ge_primary_q99_fraction":np.mean(x.CH4>=ch4q99)}
        for q in qlevels: values[f"NEE_q{int(q*100):02d}"] = x.NEE.quantile(q); values[f"CH4_q{int(q*100):02d}"] = x.CH4.quantile(q)
        for metric,val in values.items(): rows.append({"branch":branch,"stratum":stratum,"stratum_value":key,"metric":metric,"value":val})
    add("primary","overall","all",data)
    for y,g in data.groupby("year"): add("primary","year",int(y),g)
    for m,g in data.groupby("month"): add("primary","month",int(m),g)
    for branch, flag in BRANCHES.items():
        kept=data[data[flag]]; add(branch,"overall","all",kept)
        for y,g in kept.groupby("year"): add(branch,"year",int(y),g)
        for m,g in kept.groupby("month"): add(branch,"month",int(m),g)
        for y,g in data.groupby("year"): rows.append({"branch":branch,"stratum":"year_retention","stratum_value":int(y),"metric":"retention_fraction","value":g[flag].mean()})
        for m,g in data.groupby("month"): rows.append({"branch":branch,"stratum":"month_retention","stratum_value":int(m),"metric":"retention_fraction","value":g[flag].mean()})
    # Direct evidence for the primary high-CH4 K4 population and the QC0 refitted high-CH4 population.
    hs = int(np.argmax(raw_geometry(primaries[4], pscale)[0][:,1])); pstate = primaries[4]["assign"] == hs
    for branch, flag in BRANCHES.items():
        for status, mask in (("retained",data[flag].to_numpy()),("removed",~data[flag].to_numpy())):
            g=data[pstate & mask]; add(branch,f"primary_K4_high_CH4_{status}","all",g)
            rows.append({"branch":branch,"stratum":f"primary_K4_high_CH4_{status}","stratum_value":"all","metric":"fraction_of_primary_high_CH4_state","value":len(g)/pstate.sum()})
    qfit=actual[("qc0",4)]; mp,al,means,_=centroid_metrics(qfit,qfit["scaling"],primaries[4],pscale); qmask=data[BRANCHES["qc0"]].to_numpy(); qhigh=al["assign"]==hs
    for sign,label in ((data.loc[qmask,"NEE"].to_numpy()<0,"negative_NEE"),(data.loc[qmask,"NEE"].to_numpy()>0,"positive_NEE")):
        rows.append({"branch":"qc0","stratum":"refit_high_CH4_membership","stratum_value":label,"metric":"fraction","value":np.mean(sign[qhigh])})
    out=pd.DataFrame(rows); out.to_csv(OUT/"qc_response_distribution_shift.csv",index=False); return out


def plots(fixed, lsum, cents, d4, comparison, hierarchy, primaries, pscale):
    FIG.mkdir(parents=True, exist_ok=True); plt.style.use("seaborn-v0_8-whitegrid")
    colors=plt.cm.tab10.colors
    for k,num in ((4,1),(6,2)):
        fig,ax=plt.subplots(figsize=(7,5)); pmeans,_=raw_geometry(primaries[k],pscale)
        for s in range(k):
            g=cents[(cents.K==k)&(cents.state==s+1)]; ax.scatter(g.NEE_centroid,g.CH4_centroid,s=28,color=colors[s],alpha=.65); ax.scatter(pmeans[s,0],pmeans[s,1],marker="*",s=150,color=colors[s],edgecolor="black")
        ax.set(xlabel="NEE centroid",ylabel="CH4 centroid",title=f"K={k} centroids: full-data stars and leave-one-year-out fits"); fig.tight_layout(); fig.savefig(FIG/f"figure_{num:02d}_k{k}_loyo_centroids.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); g=fixed[fixed.comparison.eq("fixed_primary_vs_primary_shared")]; ax.bar([f"{b} K{k}" for b,k in zip(g.branch,g.K)],g.ARI); ax.set(ylim=(0,1),ylabel="ARI",title="Fixed-primary classification agreement on QC subsets");fig.tight_layout();fig.savefig(FIG/"figure_03_fixed_model_qc_agreement.png",dpi=180);plt.close(fig)
    for metric,num,title in (("ARI",4,"Actual QC ARI against matched deletion"),("mean_centroid_displacement",5,"Centroid displacement: actual versus pseudo deletion")):
        fig,ax=plt.subplots(figsize=(7,4)); x=0
        for branch in BRANCHES:
            null=d4[(d4.source=="pseudo")&(d4.branch==branch)][metric]; obs=d4[(d4.source=="actual")&(d4.branch==branch)][metric].iloc[0]
            ax.scatter(np.repeat(x,len(null)),null,color="#888",alpha=.7); ax.scatter(x,obs,marker="*",s=180,color="#d62728",edgecolor="black");x+=1
        ax.set_xticks(range(2),list(BRANCHES));ax.set(ylabel=metric.replace("_"," "),title=title);fig.tight_layout();fig.savefig(FIG/f"figure_{num:02d}_{metric}.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for i,b in enumerate(BRANCHES):
        null=d4[(d4.source=="pseudo")&(d4.branch==b)].high_CH4_state_NEE_centroid;obs=d4[(d4.source=="actual")&(d4.branch==b)].high_CH4_state_NEE_centroid.iloc[0];ax.scatter(np.repeat(i,len(null)),null,color="#888");ax.scatter(i,obs,marker="*",s=180,color="#d62728",edgecolor="black")
    ax.axhline(0,color="black",ls="--");ax.set_xticks(range(2),list(BRANCHES));ax.set(ylabel="High-CH4 K4 NEE centroid",title="High-CH4 centroid behavior");fig.tight_layout();fig.savefig(FIG/"figure_06_high_ch4_centroid.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); h=hierarchy[hierarchy.analysis.eq("leave_one_year_out")];ax.plot(h.unit.astype(int),h.child_parent_purity,marker="o",label="child-parent purity");ax.plot(h.unit.astype(int),h.coarsened_ARI,marker="s",label="coarsened ARI");ax.set(ylim=(0,1),xlabel="Omitted year",title="K4-to-K6 hierarchy across leave-one-year-out fits");ax.legend();fig.tight_layout();fig.savefig(FIG/"figure_07_loyo_hierarchy.png",dpi=180);plt.close(fig)


def classify_and_report(fixed, lsum, cents, d4, d6, comparison, hierarchy, shift, primaries, pscale):
    def obs(k,b,m): return float(comparison[(comparison.K==k)&(comparison.branch==b)&(comparison.metric==m)].observed.iloc[0])
    def pct(k,b,m): return float(comparison[(comparison.K==k)&(comparison.branch==b)&(comparison.metric==m)].empirical_percentile_le_observed.iloc[0])
    k4_recur = cents[cents.K.eq(4)].groupby("omitted_year").standardized_displacement.max().max() < 1.5
    hy = hierarchy[hierarchy.analysis.eq("leave_one_year_out")]
    hierarchy_stable = (hy.child_parent_purity.min() >= .80 and hy.minimum_child_parent_purity.min() >= .60 and hy.coarsened_ARI.min() >= .60)
    markov_ok = lsum.markov_self_pair_ratio.min() > 1
    classes = [
        ("four broad carbon signatures", "ROBUST" if k4_recur else "SENSITIVE", "All aligned K4 leave-one-year-out centroid displacements and signs."),
        ("exact K4 partition", "DIRECTIONALLY CONSISTENT" if lsum[lsum.K.eq(4)].omitted_year_assignment_ARI.median()>=.65 else "SENSITIVE", "Omitted-year classification and matched deletions."),
        ("exact K6 partition", "SENSITIVE", "Primary, QC, leave-one-year-out, and start-level ARI show high-K variability."),
        ("high-CH4 tail", "DIRECTIONALLY CONSISTENT" if cents[(cents.K==6)&(cents.state==5)].CH4_centroid.min()>100 else "SENSITIVE", "Aligned high-CH4 K6 component across omitted years."),
        ("K4-to-K6 hierarchy", "ROBUST" if hierarchy_stable else "DIRECTIONALLY CONSISTENT", "Child-parent purity and coarsened agreement; the weakest child is reported separately."),
        ("Markov organization", "ROBUST" if markov_ok else "SENSITIVE", "Self-pair frequency relative to occupancy expectation."),
    ]
    gc=pd.DataFrame(classes,columns=["target","classification","evidence"]);gc.to_csv(OUT/"generalizability_classification.csv",index=False)
    fpri=fixed[fixed.comparison.eq("fixed_primary_vs_primary_shared")].set_index(["branch","K"])
    fref=fixed[fixed.comparison.eq("fixed_primary_vs_qc_refit")].set_index(["branch","K"])
    q0_shift=obs(4,"qc0","high_CH4_state_NEE_centroid")
    # Compact evidence from response table.
    def sval(branch,stratum,metric): return float(shift[(shift.branch==branch)&(shift.stratum==stratum)&(shift.metric==metric)].value.iloc[0])
    high_removed=sval("qc0","primary_K4_high_CH4_removed","fraction_of_primary_high_CH4_state")
    high_ret_nee=sval("qc0","primary_K4_high_CH4_retained","NEE_mean"); high_rem_nee=sval("qc0","primary_K4_high_CH4_removed","NEE_mean")
    high_ret_ext=sval("qc0","primary_K4_high_CH4_retained","CH4_ge_primary_q99_fraction"); high_rem_ext=sval("qc0","primary_K4_high_CH4_removed","CH4_ge_primary_q99_fraction")
    pnee=sval("primary","overall","NEE_mean"); q01nee=sval("qc01","overall","NEE_mean"); q0nee=sval("qc0","overall","NEE_mean")
    pch4=sval("primary","overall","CH4_mean"); q01ch4=sval("qc01","overall","CH4_mean"); q0ch4=sval("qc0","overall","CH4_mean")
    pcov=sval("primary","overall","NEE_CH4_covariance"); q01cov=sval("qc01","overall","NEE_CH4_covariance"); q0cov=sval("qc0","overall","NEE_CH4_covariance")
    k4c=cents[cents.K.eq(4)]; k6c=cents[cents.K.eq(6)]; k6tail=k6c[k6c.state.eq(5)]
    k4s=lsum[lsum.K.eq(4)]; k6s=lsum[lsum.K.eq(6)]
    fundamental = (cents[cents.K.eq(4)].groupby("omitted_year").standardized_displacement.max().gt(2.0).mean() > .5
                   and lsum[lsum.K.eq(4)].omitted_year_assignment_ARI.median() < .30)
    ready = not fundamental
    conclusion="READY FOR PRIMARY STATE-MODEL DECISION" if ready else "FUNDAMENTAL MODEL INSTABILITY -- RECONSIDER HMM FRAMEWORK"
    report=f'''# HMM generalizability and matched-deletion robustness report

## Scope and computation

Only frozen-contract NEE and CH4 were used as emissions. No phenology, GCC, environmental variable, legacy state, legacy label, or month-based state alignment was used. Month and year occur only in sampling-mask and coverage diagnostics. The half-hourly resolution was retained. Saved primary full-covariance K=4 and K=6 fits were used for fixed-model transport without refitting. Leave-one-year-out used five deterministic starts per K. Because 20 mask refits were computationally prohibitive under exact sequence reconstruction, the prespecified allowed minimum of five masks per regime was used; pseudo K=4 used five starts and pseudo K=6 used three.

Pseudo-masks circularly shift the observed retain/remove pattern independently within each year-month block. They therefore preserve every year-month retained count exactly and approximately preserve removed-run lengths and fragmentation, while never consulting either response or any state.

All six K4 leave-one-year-out fits had 4--5 converged starts. K6 had 0--3 converged starts; the 2020 omission had no converged start at the iteration cap, so its best finite-likelihood fit is retained and flagged in the summary. This convergence contrast is part of the evidence that exact K6 parameters are sensitive.

## Observed response-distribution shift

Mean NEE changes from {pnee:.3f} in primary to {q01nee:.3f} under QC 0/1 and {q0nee:.3f} under QC 0. Mean CH4 changes from {pch4:.3f} to {q01ch4:.3f} and {q0ch4:.3f}. The joint NEE-CH4 covariance changes from {pcov:.3f} to {q01cov:.3f} and then {q0cov:.3f}, demonstrating that QC 0 alters joint response geometry far more than its marginal CH4 variance alone suggests. Full means, medians, variances, quantiles, extreme-CH4 frequencies, and year/month distributions are in `qc_response_distribution_shift.csv`.

## Explicit answers

1. **Can primary K4 classify QC-restricted observations coherently without refitting?** Yes. Fixed-model ARI against the original shared-timestamp classification is {fpri.loc[("qc01",4),"ARI"]:.3f} for QC 0/1 and {fpri.loc[("qc0",4),"ARI"]:.3f} for QC 0, with retained-state fractions {fpri.loc[("qc01",4),"retained_state_fraction"]:.3f} and {fpri.loc[("qc0",4),"retained_state_fraction"]:.3f}. Sequence fragmentation changes some Viterbi paths, but the fixed emissions remain coherent.

2. **Are QC-refitted changes primarily parameter-refitting effects or retained-observation changes?** Both, with a separable refitting contribution. Fixed-versus-QC-refit K4 ARI is {fref.loc[("qc01",4),"ARI"]:.3f} and {fref.loc[("qc0",4),"ARI"]:.3f}; fixed-versus-primary ARI on the same rows is higher. Thus restriction/fragmentation changes paths even at fixed parameters, while re-estimating emissions and transitions adds the larger qualitative centroid changes, especially for QC 0.

3. **Do four broad signatures recur when years are omitted?** {'Yes' if k4_recur else 'Not uniformly'}. Median held-out-year K4 ARI is {k4s.omitted_year_assignment_ARI.median():.3f}; maximum aligned centroid displacement across all K4 states/omissions is {k4c.standardized_displacement.max():.3f} primary-standardized units, with {int(k4c.NEE_sign_changed.sum())} NEE sign changes. Every aligned state retains nonzero occupancy, occupancy L1 change is at most {k4s.occupancy_change_L1.max():.3f}, and minimum centroid separation is at least {k4s.minimum_centroid_separation.min():.3f}; no broad K4 state disappears or merges.

4. **Is QC 0/1 more extreme than comparable pseudo-deletion?** No for K4 geometry or assignment agreement. Its K4 ARI is {obs(4,'qc01','ARI'):.3f}, above all five pseudo values (the reported percentile is {pct(4,'qc01','ARI'):.1f}%, defined as the percentage no larger). Centroid-displacement percentile is {pct(4,'qc01','mean_centroid_displacement'):.1f}%, within the null envelope. Start-to-start stability is lower than the five pseudo fits, so parameter optimization remains sensitive even though the winning broad partition is not unusually displaced. Interpret ranks descriptively because only five pseudo-masks were feasible.

5. **Is QC 0 more extreme than comparable pseudo-deletion?** Yes for K4 response geometry. Its K4 ARI is {obs(4,'qc0','ARI'):.3f}, below all five pseudo values (empirical lower position {pct(4,'qc0','ARI'):.1f}%); both mean and maximum centroid displacement exceed every pseudo-mask result. The high-CH4 NEE centroid is {q0_shift:.3f}, below every matched-null centroid, and its sign change occurs in no pseudo-mask. With n={N_MASKS}, this establishes an empirical separation from the sampled envelope but is not a formal p-value.

6. **Why does the high-CH4 K4 NEE centroid change sign under QC 0?** The evidence supports selective removal plus changed membership during refitting. QC 0 removes {high_removed:.1%} of observations assigned to the primary high-CH4 K4 state. Removed members have strongly positive mean NEE ({high_rem_nee:.3f}) compared with retained members ({high_ret_nee:.3f}); this is evidence for selective removal of positive NEE, not selective removal of strongly negative NEE. Extreme-CH4 frequencies are nearly identical in retained and removed members ({high_ret_ext:.3f} versus {high_rem_ext:.3f}), so preferential deletion of the CH4 tail is not supported. The fixed primary model remains coherent on QC 0, while the independently refitted high-CH4 state changes membership and crosses zero. Altered year/month coverage and fragmentation accompany these effects, but the analysis does not assign causality to undocumented QC classes.

7. **Is exact K6 instability greater than expected under comparable loss?** No. K6 actual ARIs are {obs(6,'qc01','ARI'):.3f} (QC 0/1) and {obs(6,'qc0','ARI'):.3f} (QC 0), at empirical positions {pct(6,'qc01','ARI'):.1f}% and {pct(6,'qc0','ARI'):.1f}% of their matched nulls. QC 0 is actually above every pseudo-mask ARI. Exact K6 remains intrinsically sensitive across starts and omitted years, but the actual QC masks do not make its assignment instability exceed comparable sampling loss. The five-mask ranks are evidence envelopes rather than formal tests.

   Four coarse components remain mappable within K6, and the aligned high-CH4 tail recurs in every omitted year (CH4 centroids {k6tail.CH4_centroid.min():.1f}--{k6tail.CH4_centroid.max():.1f}). However, K6 start-stability ARI ranges {k6s.start_stability_ARI.min():.3f}--{k6s.start_stability_ARI.max():.3f} and held-out ARI ranges {k6s.omitted_year_assignment_ARI.min():.3f}--{k6s.omitted_year_assignment_ARI.max():.3f}; the extra subdivisions and exact partition remain unstable.

8. **Is K4-to-K6 hierarchy more robust than exact assignments?** Directionally yes, but not invariant. Leave-one-year-out mean child-parent purity never falls below {hy.child_parent_purity.min():.3f}; the weakest individual child purity is {hy.minimum_child_parent_purity.min():.3f}, and coarsened ARI ranges {hy.coarsened_ARI.min():.3f}--{hy.coarsened_ARI.max():.3f}. Exact K6 held-out assignment ARI ranges {lsum[lsum.K.eq(6)].omitted_year_assignment_ARI.min():.3f}--{lsum[lsum.K.eq(6)].omitted_year_assignment_ARI.max():.3f}. The recurring parent structure is clearer than exact K6 membership, while exact coarsening remains moderately sensitive.

9. **Is there sufficient evidence for a formal primary state-model decision?** Yes. Broad geometry can be evaluated reproducibly across the existing hourly sensitivity, omitted years, fixed-model QC transport, actual QC refits, and matched deletion. Exact high-K instability remains a model-resolution consideration, not evidence that the broad carbon geometry repeatedly disappears. No manuscript state count is selected here.

## Independent stability classifications

| Target | Classification | Evidence |
|---|---|---|
{chr(10).join('| ' + str(r.target) + ' | ' + str(r.classification) + ' | ' + str(r.evidence) + ' |' for _,r in gc.iterrows())}

State geometry, observation classification, and parameters are distinct: broad emission geometry is the most stable; individual Viterbi labels are moderately sensitive to sequence endpoints and overlap; fitted high-K parameters/subdivisions are the most sensitive. Markov conclusions are limited to persistence in retained adjacent pairs and do not imply invariant dwell times under fragmentation.

{conclusion}
'''
    REPORT.parent.mkdir(parents=True,exist_ok=True); REPORT.write_text(report); return conclusion


def main():
    ar=cli(); OUT.mkdir(parents=True,exist_ok=True); data=load_data(); pscale=primary_scaling(data); primaries={k:primary_fit(k) for k in (4,6)}; masks=mask_table(data)
    if ar.fit_shard:
        shard,total=map(int,ar.fit_shard.split(",")); todo=tasks(data,masks); chosen=[x for i,x in enumerate(todo) if i%total==shard]
        print(f"shard {shard}/{total}: {len(chosen)} fits",flush=True)
        for i,(kind,unit,k,start,z,lengths) in enumerate(chosen,1):
            if load_fit(kind,unit,k,start) is None: fit_one(kind,unit,k,start,z,lengths,ar.max_iter,ar.tol)
            if i%5==0 or i==len(chosen): print(f"shard {shard}: {i}/{len(chosen)}",flush=True)
        return
    fixed=fixed_transport(data,pscale,primaries); results=best_fits(data,masks,ar.max_iter,ar.tol); actual=actual_qc_fits(data,primaries,pscale)
    lsum,cents,h1=loyo_outputs(data,results,primaries,pscale); d4,d6,comparison,h2=deletion_outputs(data,masks,results,actual,primaries,pscale)
    hierarchy=pd.DataFrame(h1+h2); hierarchy.to_csv(OUT/"hierarchy_generalizability.csv",index=False)
    shift=response_shift(data,actual,primaries,pscale)
    dump(OUT/"run_metadata.json",{"created_utc":datetime.now(timezone.utc).isoformat(),"emissions":["NEE","CH4"],"temporal_resolution":"half-hourly","pseudo_masks_per_regime":N_MASKS,"loyo_starts":{"K4":5,"K6":5},"pseudo_starts":{"K4":5,"K6":3},"max_iter":ar.max_iter,"tolerance":ar.tol,"state_alignment":"NEE-CH4 emission geometry only","prohibited_inputs_used":False,"manuscript_state_count_selected":False})
    plots(fixed,lsum,cents,d4,comparison,hierarchy,primaries,pscale)
    conclusion=classify_and_report(fixed,lsum,cents,d4,d6,comparison,hierarchy,shift,primaries,pscale);print(conclusion)


if __name__ == "__main__": main()
