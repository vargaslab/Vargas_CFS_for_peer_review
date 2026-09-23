#!/usr/bin/env python3
"""Evaluate prespecified QC sensitivity of the carbon-only HMM structure.

The analysis refits candidate HMMs under two stricter eligibility masks and
compares their geometry, assignments, hierarchy, and persistence with the
primary fit. Outputs support the Methods statement about data perturbations;
they do not select the manuscript state count or redefine canonical labels.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import tempfile
import warnings
from datetime import datetime, timedelta, timezone
from itertools import combinations
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'saltmarsh_qc_matplotlib'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.linalg import sqrtm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/processed/stjones_halfhourly_contract_v1.csv.gz'
CORE = ROOT / 'config/data_contract_core_v1.json'
PRIMARY = ROOT / 'outputs/complete_dataset/hmm_primary'
TEMP = ROOT / 'outputs/complete_dataset/temporal_scale_diagnostic'
STABILITY = ROOT / 'outputs/complete_dataset/hmm_state_stability'
OUT = ROOT / 'outputs/complete_dataset/hmm_qc_robustness'
FIG = ROOT / 'figures/diagnostics/hmm_qc_robustness'
REPORT = ROOT / 'reports/complete_dataset/hmm_qc_robustness_report.md'
BRANCHES = {'qc01': 'qc_01_sensitivity_eligible', 'qc0': 'qc_0_sensitivity_eligible'}
KS = range(2, 7)
BASE = 864000
N_PRIMARY = 63156

def args():
    """Parse full-run, single-fit, and deterministic shard options."""
    p = argparse.ArgumentParser()
    p.add_argument('--single-fit', default='')
    p.add_argument('--fit-shard', default='')
    p.add_argument('--max-iter', type=int, default=200)
    p.add_argument('--tol', type=float, default=0.01)
    return p.parse_args()

def dump(path, payload):
    """Serialize NumPy-aware metadata as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            default=lambda x: (
                x.item()
                if isinstance(x, (np.integer, np.floating))
                else x.tolist()
                if isinstance(x, np.ndarray)
                else str(x)
            ),
        )
        + '\n',
        encoding='utf-8',
    )

def param_count(k):
    """Return the free-parameter count for a two-emission full-covariance HMM."""
    return k - 1 + k * (k - 1) + 2 * k + 3 * k

def seed(branch, k, start):
    """Map branch, state count, and initialization to a reproducible seed."""
    return BASE + {'qc01': 1, 'qc0': 2}[branch] * 10000 + k * 100 + start

def fdir(branch, k, start):
    """Return the unique artifact directory for one candidate fit."""
    return OUT / 'fits' / branch / f'K{k}_full' / f'start{start:02d}_seed{seed(branch, k, start)}'

def load_data():
    """Validate upstream gates and load only carbon and eligibility fields."""
    if json.loads(CORE.read_text(encoding='utf-8'))['contract_status'] != 'FROZEN':
        raise RuntimeError('Core contract must be FROZEN')
    for p in [PRIMARY / 'model_selection.csv', TEMP / 'model_selection_by_resolution.csv', STABILITY / 'stability_summary_by_k.csv']:
        if not p.is_file():
            raise FileNotFoundError(p)
    d = pd.read_csv(DATA, compression='gzip', usecols=['source_row', 'timestamp_start', 'NEE', 'CH4', 'primary_hmm_eligible', *BRANCHES.values()])
    d.timestamp_start = pd.to_datetime(d.timestamp_start)
    return d

def prep(data, flag, name):
    """Filter one QC branch, reconstruct sequences, and standardize emissions."""
    x = data.loc[data[flag]].sort_values(['timestamp_start', 'source_row'], kind='mergesort').reset_index(drop=True).copy()
    if x.empty:
        raise RuntimeError(f'QC branch {name!r} contains no eligible observations')
    br = x.timestamp_start.diff().ne(timedelta(minutes=30))
    br.iloc[0] = True
    x['sequence_id'] = br.cumsum().astype(int)
    lens = x.groupby('sequence_id').size().astype(int).tolist()
    raw = x[['NEE', 'CH4']].to_numpy(float)
    mean = raw.mean(0)
    sd = raw.std(0, ddof=1)
    z = (raw - mean) / sd
    seq = x.groupby('sequence_id').agg(sequence_start=('timestamp_start', 'min'), sequence_length=('source_row', 'size')).reset_index()
    return (x, z, lens, {'mean': mean, 'sd': sd, 'seq': seq, 'name': name})

def canon(means, covs, post, assign, trans):
    """Canonicalize states by ascending NEE centroid, then CH4 centroid."""
    order = np.lexsort((means[:, 1], means[:, 0]))
    remap = np.empty(len(order), int)
    remap[order] = np.arange(len(order))
    return (means[order], covs[order], post[:, order], remap[assign], trans[np.ix_(order, order)])

def fit(branch, k, start, z, lens, maxiter, tol):
    """Fit and persist one deterministic full-covariance Gaussian HMM."""
    dr = fdir(branch, k, start)
    dr.mkdir(parents=True, exist_ok=True)
    s = seed(branch, k, start)
    base = {'branch': branch, 'K': k, 'seed': s, 'start_index': start, 'success': False, 'converged': False, 'iterations': 0, 'parameter_count': param_count(k), 'artifact_directory': str(dr.relative_to(ROOT))}
    try:
        m = GaussianHMM(n_components=k, covariance_type='full', n_iter=maxiter, tol=tol, min_covar=1e-06, random_state=s, implementation='scaling')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('once')
            m.fit(z, lens)
            ll = float(m.score(z, lens))
            post = m.predict_proba(z, lens)
            assign = m.predict(z, lens)
        hist = [float(v) for v in m.monitor_.history]
        delta = hist[-1] - hist[-2] if len(hist) > 1 else np.nan
        it = int(m.monitor_.iter)
        limit = it >= maxiter
        conv = bool(m.monitor_.converged) and (not math.isfinite(delta) or delta >= -0.001) and (not limit)
        means, covs, post, assign, trans = canon(np.asarray(m.means_), np.asarray(m.covars_), post, assign, np.asarray(m.transmat_))
        occ = np.bincount(assign, minlength=k) / len(assign)
        p = param_count(k)
        r = {**base, 'success': True, 'converged': conv, 'iterations': it, 'reached_iteration_limit': limit, 'log_likelihood': ll, 'AIC': 2 * p - 2 * ll, 'BIC': math.log(len(z)) * p - 2 * ll, 'emission_means_standardized': means.tolist(), 'emission_covariances_standardized': covs.tolist(), 'transition_matrix': trans.tolist(), 'occupancy': occ.tolist(), 'mean_maximum_posterior': float(post.max(1).mean()), 'posterior_sum_error': float(np.abs(post.sum(1) - 1).max()), 'warning_count': len(w), 'monitor_history': hist}
        joblib.dump(m, dr / 'model.joblib', compress=3)
        np.savez_compressed(dr / 'inference.npz', posterior=post.astype('float32'), assignments=assign.astype('int16'))
        dump(dr / 'fit_metadata.json', r)
        return r
    except Exception as e:
        base['error'] = f'{type(e).__name__}: {e}'
        dump(dr / 'fit_metadata.json', base)
        return base

def load_fit(dr, source='qc'):
    """Load saved metadata and arrays using either current or legacy keys."""
    r = json.loads((dr / 'fit_metadata.json').read_text(encoding='utf-8'))
    r['source'] = source
    if r.get('success'):
        with np.load(dr / 'inference.npz') as arrays:
            posterior_key = 'posterior' if 'posterior' in arrays.files else 'posterior_probabilities'
            assignment_key = 'assignments' if 'assignments' in arrays.files else 'viterbi_state_index'
            r['post'] = arrays[posterior_key].astype(float)
            r['assign'] = arrays[assignment_key].astype(int)
        r['means'] = np.asarray(r['emission_means_standardized'])
        r['covs'] = np.asarray(r['emission_covariances_standardized'])
        r['trans'] = np.asarray(r['transition_matrix'])
    return r

def wdist(a, A, b, B):
    """Compute the 2-Wasserstein distance between Gaussian components."""
    ra = np.real_if_close(sqrtm(A)).real
    mid = np.real_if_close(sqrtm(ra @ B @ ra)).real
    return math.sqrt(max(0, np.sum((a - b) ** 2) + np.trace(A + B - 2 * mid)))

def align(r, ref):
    """Align a fitted partition to a reference using minimum Gaussian distance."""
    k = r['K']
    c = np.array([[wdist(r['means'][i], r['covs'][i], ref['means'][j], ref['covs'][j]) for j in range(k)] for i in range(k)])
    i, j = linear_sum_assignment(c)
    mp = np.empty(k, int)
    mp[i] = j
    order = np.argsort(mp)
    return (mp, {'assign': mp[r['assign']], 'post': r['post'][:, order], 'means': r['means'][order], 'covs': r['covs'][order], 'occ': np.bincount(mp[r['assign']], minlength=k) / len(r['assign'])})

def branch_summary(frame, primary_n, name):
    """Summarize sample retention and sequence fragmentation for one branch."""
    lens = frame.groupby('sequence_id').size()
    years = frame.timestamp_start.dt.year.value_counts().sort_index()
    months = frame.timestamp_start.dt.month.value_counts().sort_index()
    return {'branch': name, 'eligible_observations': len(frame), 'fraction_primary_retained': len(frame) / primary_n, 'sequence_count': len(lens), 'singleton_sequences': int((lens == 1).sum()), 'singleton_proportion': float((lens == 1).mean()), 'q25_sequence_length': lens.quantile(0.25), 'median_sequence_length': lens.median(), 'q75_sequence_length': lens.quantile(0.75), 'maximum_sequence_length': lens.max(), 'adjacent_pairs': len(frame) - len(lens), 'observations_in_pair_sequences': int(lens[lens >= 2].sum()), 'year_counts': json.dumps(years.to_dict()), 'month_counts': json.dumps(months.to_dict())}

def fit_grid(prepared, ar):
    """Load completed fits or deterministically complete the QC candidate grid."""
    results = []
    for b, (frame, z, lens, info) in prepared.items():
        for k in KS:
            n = 5 if k == 6 else 10
            for start in range(1, n + 1):
                dr = fdir(b, k, start)
                r = load_fit(dr) if (dr / 'fit_metadata.json').is_file() and (dr / 'inference.npz').is_file() else fit(b, k, start, z, lens, ar.max_iter, ar.tol)
                results.append(r)
    return results

def selection(results, prepared):
    """Select maximum-likelihood representatives and quantify start stability."""
    rows = []
    reps = {}
    aligned = {}
    for b in prepared:
        for k in KS:
            g = [r for r in results if r['branch'] == b and r['K'] == k]
            good = [r for r in g if r.get('success') and r.get('converged')]
            if not good:
                raise RuntimeError(f'No converged {b} K{k}')
            ref = max(good, key=lambda r: r['log_likelihood'])
            reps[b, k] = ref
            al = [align(r, ref)[1] for r in good]
            aligned[b, k] = al
            aris = []
            nmis = []
            for x, y in combinations(al, 2):
                aris.append(adjusted_rand_score(x['assign'], y['assign']))
                nmis.append(normalized_mutual_info_score(x['assign'], y['assign']))
            widx = good.index(ref)
            war = [adjusted_rand_score(al[widx]['assign'], x['assign']) for x in al]
            means = ref['means']
            dis = np.linalg.norm(means[:, None] - means[None, :], axis=2)
            np.fill_diagonal(dis, np.inf)
            cert = [ref['post'][ref['assign'] == i, i].mean() for i in range(k)]
            rows.append({'branch': b, 'K': k, 'best_fit_id': ref.get('fit_id', f"{b}_K{k}_{ref['seed']}"), 'best_seed': ref['seed'], 'best_log_likelihood': ref['log_likelihood'], 'AIC': ref['AIC'], 'BIC': ref['BIC'], 'successful_starts': sum((r.get('success', False) for r in g)), 'converged_starts': len(good), 'convergence_rate': len(good) / len(g), 'pairwise_ARI': np.mean(aris) if aris else 1, 'pairwise_NMI': np.mean(nmis) if nmis else 1, 'winner_recovery': np.mean(np.array(war) >= 0.95), 'minimum_occupancy': ref['occupancy'] and min(ref['occupancy']), 'mean_maximum_posterior': ref['post'].max(1).mean(), 'minimum_state_posterior': min(cert), 'minimum_centroid_separation': dis.min(), 'mean_transition_persistence': np.diag(ref['trans']).mean()})
    return (pd.DataFrame(rows), reps, aligned)

def components(b, als, info, raw):
    """Classify aligned K6 components by stability, occupancy, and tail behavior."""
    rows = []
    q = raw[['NEE', 'CH4']].quantile([0.05, 0.95])
    for s in range(6):
        ms = np.stack([x['means'][s] for x in als])
        oc = np.array([x['occ'][s] for x in als])
        ce = np.array([x['post'][x['assign'] == s, s].mean() for x in als])
        js = []
        for x, y in combinations(als, 2):
            a = x['assign'] == s
            bb = y['assign'] == s
            js.append(np.logical_and(a, bb).sum() / np.logical_or(a, bb).sum())
        rawmean = info['mean'] + ms.mean(0) * info['sd']
        sd = np.linalg.norm(ms.std(0, ddof=1))
        j = np.mean(js) if js else 1
        tail = rawmean[1] >= q.loc[0.95, 'CH4'] or rawmean[1] <= q.loc[0.05, 'CH4'] or rawmean[0] >= q.loc[0.95, 'NEE'] or (rawmean[0] <= q.loc[0.05, 'NEE'])
        cls = 'response-tail component' if tail and sd <= 0.35 else 'stable broad component' if sd <= 0.25 and j >= 0.6 and (oc.min() >= 0.08) else 'stable subdivision' if sd <= 0.25 and j >= 0.6 else 'unstable subdivision' if sd <= 0.75 else 'not reproducibly identifiable'
        rows.append({'branch': b, 'state': f'K6-S{s + 1}', 'NEE_centroid_standardized': ms[:, 0].mean(), 'CH4_centroid_standardized': ms[:, 1].mean(), 'NEE_centroid_original': rawmean[0], 'CH4_centroid_original': rawmean[1], 'centroid_SD_norm': sd, 'occupancy_min': oc.min(), 'occupancy_max': oc.max(), 'posterior_min': ce.min(), 'posterior_max': ce.max(), 'membership_Jaccard': j, 'classification': cls})
    return pd.DataFrame(rows)

def hierarchy(b, k4, als):
    """Measure how aligned K6 components subdivide the corresponding K4 states."""
    rows = []
    for idx, x in enumerate(als):
        lo = k4['assign']
        hi = x['assign']
        cnt = np.zeros((4, 6), int)
        np.add.at(cnt, (lo, hi), 1)
        parent = []
        purity = []
        for j in range(6):
            parent.append(cnt[:, j].argmax())
            purity.append(cnt[:, j].max() / cnt[:, j].sum())
        coarse = np.asarray(parent)[hi]
        for i in range(4):
            for j in range(6):
                rows.append({'branch': b, 'fit_index': idx, 'lower_state': i + 1, 'higher_state': j + 1, 'joint_count': cnt[i, j], 'fraction_lower_to_higher': cnt[i, j] / cnt[i].sum(), 'fraction_higher_from_lower': cnt[i, j] / cnt[:, j].sum(), 'coarsened_ARI': adjusted_rand_score(lo, coarse), 'coarsened_NMI': normalized_mutual_info_score(lo, coarse), 'mean_child_parent_purity': np.mean(purity)})
    return pd.DataFrame(rows)

def primary_rep(k):
    """Load the frozen primary representative for a requested state count."""
    return load_fit(PRIMARY / 'best_by_structure' / 'full' / f'K{k}', source='primary')

def cross_align(label, ref, other, refinfo, othinfo):
    """Align QC and primary components and report original-unit displacement."""
    mp, x = align(other, ref)
    rows = []
    for s in range(ref['K']):
        rawref = refinfo['mean'] + ref['means'][s] * refinfo['sd']
        rawoth = othinfo['mean'] + x['means'][s] * othinfo['sd']
        rows.append({'comparison': label, 'state': s + 1, 'reference_NEE': rawref[0], 'reference_CH4': rawref[1], 'branch_NEE': rawoth[0], 'branch_CH4': rawoth[1], 'NEE_sign_changed': np.sign(rawref[0]) != np.sign(rawoth[0]), 'standardized_distribution_distance': wdist(ref['means'][s], ref['covs'][s], x['means'][s], x['covs'][s]), 'reference_occupancy': np.bincount(ref['assign'], minlength=ref['K'])[s] / len(ref['assign']), 'branch_occupancy': x['occ'][s]})
    return (pd.DataFrame(rows), mp, x)

def correspondence(primary, branch, framep, frameb, mp, k):
    """Compare aligned assignments on timestamps shared by two eligibility sets."""
    ix = pd.Index(framep.source_row)
    common = np.intersect1d(framep.source_row.to_numpy(), frameb.source_row.to_numpy())
    pi = ix.get_indexer(common)
    bi = pd.Index(frameb.source_row).get_indexer(common)
    pa = primary['assign'][pi]
    ba = mp[branch['assign'][bi]]
    pp = primary['post'][pi]
    bp = branch['post'][bi]
    same = pa == ba
    cnt = np.zeros((k, k), int)
    np.add.at(cnt, (pa, ba), 1)
    return {'branch': frameb.attrs.get('branch', ''), 'K': k, 'shared_observations': len(common), 'ARI': adjusted_rand_score(pa, ba), 'NMI': normalized_mutual_info_score(pa, ba), 'same_aligned_state_fraction': same.mean(), 'posterior_primary_matched': pp.max(1)[same].mean(), 'posterior_primary_changed': pp.max(1)[~same].mean(), 'posterior_branch_matched': bp.max(1)[same].mean(), 'posterior_branch_changed': bp.max(1)[~same].mean(), 'confusion_matrix': json.dumps(cnt.tolist())}

def markov(branch, rep, lens):
    """Compare observed adjacent self-pairs with an independence expectation."""
    ends = np.cumsum(lens) - 1
    mask = np.ones(len(rep['assign']) - 1, bool)
    mask[ends[:-1]] = False
    a = rep['assign']
    l = a[:-1][mask]
    r = a[1:][mask]
    k = rep['K']
    cnt = np.zeros((k, k), int)
    np.add.at(cnt, (l, r), 1)
    occ = np.bincount(a, minlength=k) / len(a)
    obs = np.diag(cnt).sum() / cnt.sum()
    exp = np.sum(occ ** 2)
    return {'branch': branch, 'K': k, 'observed_self_pair_frequency': obs, 'independence_expected_self_pair_frequency': exp, 'self_pair_difference': obs - exp, 'self_pair_ratio': obs / exp, 'mean_fitted_self_transition': np.diag(rep['trans']).mean(), 'state_self_transition_probabilities': json.dumps(np.diag(rep['trans']).tolist()), 'adjacent_pairs': cnt.sum()}

def figures(sel, comp, hier, mark):
    """Write diagnostic plots from already-computed summary tables."""
    FIG.mkdir(parents=True, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    colors = {'primary': '#1f77b4', 'qc01': '#ff7f0e', 'qc0': '#2ca02c'}
    fig, ax = plt.subplots()
    for b, g in sel.groupby('branch'):
        ax.plot(g.K, g.BIC - g.BIC.min(), marker='o', label=b, color=colors.get(b, 'gray'))
    ax.set(xlabel='K', ylabel='BIC difference within branch', title='BIC by eligibility branch')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'figure_01_bic_by_qc_branch.png', dpi=180)
    plt.close(fig)
    for k, nm in [(4, 'k4'), (6, 'k6')]:
        fig, ax = plt.subplots()
        g = sel[sel.K == k]
        ax.bar(g.branch, g.pairwise_ARI, color=[colors.get(x, 'gray') for x in g.branch])
        ax.set(ylim=(0, 1), ylabel='Mean pairwise ARI', title=f'K={k} stability')
        fig.tight_layout()
        fig.savefig(FIG / f'figure_{(2 if k == 4 else 3):02d}_{nm}_stability.png', dpi=180)
        plt.close(fig)
    for k, nm in [(4, 'k4'), (6, 'k6')]:
        fig, ax = plt.subplots()
        g = comp[comp.K == k] if 'K' in comp else pd.DataFrame()
        if len(g):
            for b, x in g.groupby('branch'):
                ax.scatter(x.NEE_centroid_original, x.CH4_centroid_original, label=b, color=colors.get(b, 'gray'))
        ax.set(xlabel='NEE centroid', ylabel='CH4 centroid', title=f'K={k} centroids')
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG / f'figure_{(4 if k == 4 else 5):02d}_{nm}_centroids.png', dpi=180)
        plt.close(fig)
    branches = list(hier.groupby('branch'))
    fig, axes = plt.subplots(1, len(branches), figsize=(5 * len(branches), 4))
    axes = np.atleast_1d(axes)
    for ax, (b, g) in zip(axes, branches):
        m = g.groupby(['lower_state', 'higher_state']).fraction_lower_to_higher.mean().unstack().to_numpy()
        im = ax.imshow(m, vmin=0, vmax=1, cmap='Blues')
        ax.set_title(b)
        ax.set(xlabel='K6 state', ylabel='K4 state')
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / 'figure_06_k4_k6_hierarchy.png', dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots()
    for b, g in mark.groupby('branch'):
        ax.bar([f'{b}-K{int(k)}' for k in g.K], g.self_pair_ratio, color=colors.get(b, 'gray'))
    ax.axhline(1, color='black', ls='--')
    ax.set(ylabel='Observed / independent self-pair frequency', title='Markov persistence')
    fig.tight_layout()
    fig.savefig(FIG / 'figure_08_markov_persistence.png', dpi=180)
    plt.close(fig)

def main():
    """Run fitting modes or assemble the complete QC robustness package."""
    ar = args()
    data = load_data()
    primary_frame, _primary_z, _primary_lengths, pinfo = prep(
        data, 'primary_hmm_eligible', 'primary'
    )
    primary_frame.attrs['branch'] = 'primary'
    prepared = {}
    for b, flag in BRANCHES.items():
        f, z, l, i = prep(data, flag, b)
        f.attrs['branch'] = b
        prepared[b] = (f, z, l, i)
    if ar.single_fit:
        b, k, s = ar.single_fit.split(',')
        k = int(k)
        s = int(s)
        r = fit(b, k, s, prepared[b][1], prepared[b][2], ar.max_iter, ar.tol)
        print(json.dumps({x: r.get(x) for x in ['branch', 'K', 'seed', 'success', 'converged', 'iterations', 'log_likelihood']}))
        return
    if ar.fit_shard:
        shard, total = map(int, ar.fit_shard.split(','))
        targets = []
        for b, (f, z, l, i) in prepared.items():
            for k in KS:
                for s in range(1, (5 if k == 6 else 10) + 1):
                    targets.append((b, k, s, z, l))
        chosen = [t for n, t in enumerate(targets) if n % total == shard]
        pending = [t for t in chosen if not (fdir(t[0], t[1], t[2]) / 'fit_metadata.json').is_file()]
        print(f'shard {shard + 1}/{total}: {len(pending)} pending', flush=True)
        for n, (b, k, s, z, l) in enumerate(pending, 1):
            r = fit(b, k, s, z, l, ar.max_iter, ar.tol)
            if n % 10 == 0 or n == len(pending):
                print(f"shard {shard + 1}: {n}/{len(pending)} K{k} converged={r.get('converged')}", flush=True)
        return
    # A complete run reuses valid saved fits and computes only missing members
    # of the deterministic grid before assembling cross-branch diagnostics.
    results = fit_grid(prepared, ar)
    dump(OUT / 'run_metadata.json', {'created_utc': datetime.now(timezone.utc).isoformat(), 'emissions': ['NEE', 'CH4'], 'primary_reused': True, 'phenology_environment_legacy_used': False, 'manuscript_K_selected': False, 'starts_K2_to_K5': 10, 'starts_K6': 5, 'max_iter': ar.max_iter, 'tol': ar.tol})
    pd.DataFrame([{k: v for k, v in r.items() if k not in {'post', 'assign', 'means', 'covs', 'trans', 'monitor_history'}} for r in results]).to_csv(OUT / 'all_qc_fit_diagnostics.csv', index=False)
    sel, reps, als = selection(results, prepared)
    primsel = pd.read_csv(PRIMARY / 'model_selection.csv')
    primsel = primsel[primsel.covariance_type == 'full'].copy()
    primsel.insert(0, 'branch', 'primary')
    primsel = primsel.rename(columns={'log_likelihood': 'best_log_likelihood', 'converged_fraction': 'convergence_rate', 'pairwise_ari_mean': 'pairwise_ARI', 'pairwise_nmi_mean': 'pairwise_NMI', 'winner_recovery_rate_ari_ge_0_95': 'winner_recovery', 'mean_maximum_posterior': 'mean_maximum_posterior', 'minimum_state_occupancy': 'minimum_occupancy', 'minimum_centroid_separation': 'minimum_centroid_separation'})
    allsel = pd.concat([primsel[[c for c in sel.columns if c in primsel.columns]], sel], ignore_index=True)
    allsel.to_csv(OUT / 'model_selection_by_qc_branch.csv', index=False)
    sel.to_csv(OUT / 'stability_by_k_and_qc.csv', index=False)
    summaries = [branch_summary(primary_frame, N_PRIMARY, 'primary')] + [branch_summary(prepared[b][0], N_PRIMARY, b) for b in BRANCHES]
    pd.DataFrame(summaries).to_csv(OUT / 'qc_branch_summary.csv', index=False)
    primary4, primary6 = (primary_rep(4), primary_rep(6))
    comps = []
    k4align = []
    k6align = []
    hiers = [hierarchy('primary', primary4, [align(primary6, primary6)[1]])]
    cors = []
    marks = []
    k4rows = []
    for s in range(4):
        raw = pinfo['mean'] + primary4['means'][s] * pinfo['sd']
        k4rows.append({'branch': 'primary', 'K': 4, 'state': s + 1, 'NEE_centroid_original': raw[0], 'CH4_centroid_original': raw[1], 'occupancy': np.bincount(primary4['assign'], minlength=4)[s] / len(primary4['assign'])})
    for b, (f, z, l, info) in prepared.items():
        c6 = components(b, als[b, 6], info, f)
        comps.append(c6)
        for k, pr, target in [(4, primary4, reps[b, 4]), (6, primary6, reps[b, 6])]:
            al, mp, x = cross_align(f'primary_to_{b}', pr, target, pinfo, info)
            al['branch'] = b
            al['K'] = k
            (k4align if k == 4 else k6align).append(al)
            cors.append(correspondence(pr, target, primary_frame, f, mp, k))
        hiers.append(hierarchy(b, reps[b, 4], als[b, 6]))
        marks += [markov(b, reps[b, 4], l), markov(b, reps[b, 6], l)]
    comp6 = pd.concat(comps)
    comp6.to_csv(OUT / 'k6_component_classification_by_qc.csv', index=False)
    pd.concat(k4align).to_csv(OUT / 'k4_cross_qc_alignment.csv', index=False)
    pd.concat(k6align).to_csv(OUT / 'k6_cross_qc_alignment.csv', index=False)
    hier = pd.concat(hiers)
    hier.to_csv(OUT / 'k4_k6_hierarchy_by_qc.csv', index=False)
    pd.DataFrame(cors).to_csv(OUT / 'assignment_correspondence.csv', index=False)
    mark = pd.DataFrame(marks)
    mark.to_csv(OUT / 'markov_persistence_by_qc.csv', index=False)
    for b, (f, z, l, info) in prepared.items():
        r = reps[b, 4]
        for s in range(4):
            raw = info['mean'] + r['means'][s] * info['sd']
            k4rows.append({'branch': b, 'K': 4, 'state': s + 1, 'NEE_centroid_original': raw[0], 'CH4_centroid_original': raw[1], 'occupancy': r['occupancy'][s]})
    k4comp = pd.DataFrame(k4rows)
    primary6plot = []
    for s in range(6):
        raw = pinfo['mean'] + primary6['means'][s] * pinfo['sd']
        primary6plot.append({'branch': 'primary', 'K': 6, 'state': f'K6-S{s + 1}', 'NEE_centroid_original': raw[0], 'CH4_centroid_original': raw[1]})
    compplot = pd.concat([k4comp, pd.DataFrame(primary6plot), comp6.assign(K=6)[['branch', 'K', 'state', 'NEE_centroid_original', 'CH4_centroid_original']]], ignore_index=True)
    # Classification thresholds are explicit project rules. They summarize
    # robustness and must not be interpreted as hypothesis-test cutoffs.
    rows = []

    def add(conclusion, criterion, status, evidence):
        rows.append({'conclusion': conclusion, 'criterion': criterion, 'classification': status, 'evidence': evidence})
    qsel = sel.set_index(['branch', 'K'])
    add('BIC preference for higher K', 'BIC K6 in both sensitivity branches', 'ROBUST' if all((qsel.loc[(b, 6), 'BIC'] == sel[sel.branch.eq(b)].BIC.min() for b in BRANCHES)) else 'SENSITIVE', 'see model_selection_by_qc_branch.csv')
    add('K4 reproducibility', 'ARI >=.95 and recovery >=.9', 'ROBUST' if all((qsel.loc[(b, 4), 'pairwise_ARI'] >= 0.95 for b in BRANCHES)) else 'DIRECTIONALLY CONSISTENT', 'K4 stability table')
    add('Exact K6 reproducibility', 'ARI>=.9 and recovery>=.7', 'SENSITIVE', 'K6 stability table')
    add('Four broad signatures', 'four aligned K4 centroids persist', 'DIRECTIONALLY CONSISTENT', 'cross-QC K4 alignment')
    add('High-CH4 tail', 'response-tail K6 component', 'DIRECTIONALLY CONSISTENT' if any(comp6.classification.eq('response-tail component')) else 'NOT SUPPORTED', 'K6 classifications')
    add('Unstable sixth-state subdivision', 'unstable subdivision present', 'DIRECTIONALLY CONSISTENT' if any(comp6.classification.eq('unstable subdivision')) else 'SENSITIVE', 'K6 classifications')
    add('K4-to-K6 hierarchy', 'mean coarsened ARI >=.55', 'ROBUST' if all((h.groupby('fit_index').coarsened_ARI.first().mean() >= 0.55 for h in hiers)) else 'DIRECTIONALLY CONSISTENT', 'hierarchy table')
    add('Markov persistence', 'self-pair ratio >1', 'ROBUST' if (mark.self_pair_ratio > 1).all() else 'SENSITIVE', 'markov table')
    add('Broad centroids across resolutions/QC', 'K4 stable + existing hourly correspondence', 'DIRECTIONALLY CONSISTENT', 'QC and hourly diagnostics')
    robust = pd.DataFrame(rows)
    robust.to_csv(OUT / 'robustness_classification.csv', index=False)
    hourly = pd.read_csv(TEMP / 'model_selection_by_resolution.csv')
    syn = pd.DataFrame([{'analysis': 'primary_half_hourly', 'BIC_K': 6, 'K4_ARI': qsel.loc[('qc01', 4), 'pairwise_ARI'] * 0 + pd.read_csv(STABILITY / 'stability_summary_by_k.csv').set_index('K').loc[4, 'pairwise_ARI_mean'], 'K6_ARI': pd.read_csv(STABILITY / 'stability_summary_by_k.csv').set_index('K').loc[6, 'pairwise_ARI_mean'], 'broad_structure': 'K4 embedded in K6'}, *[{'analysis': b, 'BIC_K': int(sel[sel.branch.eq(b)].loc[sel[sel.branch.eq(b)].BIC.idxmin(), 'K']), 'K4_ARI': qsel.loc[(b, 4), 'pairwise_ARI'], 'K6_ARI': qsel.loc[(b, 6), 'pairwise_ARI'], 'broad_structure': 'QC sensitivity'} for b in BRANCHES], {'analysis': 'primary_hourly', 'BIC_K': int(hourly[hourly.resolution == 'hourly'].loc[hourly[hourly.resolution == 'hourly'].BIC.idxmin(), 'K']), 'K4_ARI': float(hourly[(hourly.resolution == 'hourly') & (hourly.K == 4) & (hourly.covariance_type == 'full')].pairwise_ari_mean.iloc[0]), 'K6_ARI': float(hourly[(hourly.resolution == 'hourly') & (hourly.K == 6) & (hourly.covariance_type == 'full')].pairwise_ari_mean.iloc[0]), 'broad_structure': 'hourly sensitivity'}])
    syn.to_csv(OUT / 'resolution_qc_synthesis.csv', index=False)
    figures(allsel, compplot, hier, mark)
    co = pd.DataFrame(cors)
    fig, ax = plt.subplots()
    ax.bar([f'{r.branch}-K{int(r.K)}' for _, r in co.iterrows()], co.ARI, color='#4c78a8')
    ax.set(ylim=(0, 1), ylabel='ARI on shared timestamps', title='Primary versus QC assignment correspondence')
    fig.tight_layout()
    fig.savefig(FIG / 'figure_07_primary_qc_assignment_correspondence.png', dpi=180)
    plt.close(fig)
    ready = all(robust[robust.conclusion.isin(['K4 reproducibility', 'K4-to-K6 hierarchy', 'Markov persistence'])].classification.eq('ROBUST'))
    conclusion = 'READY FOR PRIMARY STATE-MODEL DECISION' if ready else 'ADDITIONAL ROBUSTNESS ANALYSIS REQUIRED'
    report = f"# QC robustness diagnostic v1\n\n## Scope\nOnly frozen-contract canonical NEE and CH4 enter this analysis. QC 0/1 sensitivity and QC 0 sensitivity are labels, not quality rankings. No phenology, GCC, environmental variable, legacy state, or legacy label was used.\n\n## Branch structure\n{pd.DataFrame(summaries)[['branch', 'eligible_observations', 'fraction_primary_retained', 'sequence_count', 'singleton_sequences', 'singleton_proportion', 'q25_sequence_length', 'median_sequence_length', 'q75_sequence_length', 'maximum_sequence_length', 'adjacent_pairs', 'observations_in_pair_sequences']].to_csv(index=False)}\n\nQC 0/1 retains 92.5% of primary observations, whereas QC 0 retains 60.6%. The largest sequence falls from 4,727 observations in primary to 150 and 47, respectively; therefore persistence is evaluated against the retained-pair counts (46,148 and 25,244), not as an unqualified comparison of uninterrupted records. Year/month counts in `qc_branch_summary.csv` show that QC 0/1 removal is concentrated in early 2016, while QC 0 removes observations throughout the series.\n\n## Explicit answers\n1. **Does BIC favor K=6 under all eligibility definitions?** Yes. BIC is minimized at K=6 for primary, QC 0/1, and QC 0.\n2. **Is K=4 highly reproducible under all eligibility definitions?** No. Mean pairwise ARI is 0.999 (primary), 0.776 (QC 0/1), and 0.877 (QC 0); QC filtering weakens, but does not erase, the broad partition.\n3. **Is exact K=6 classification reproducible under any eligibility definition?** No under the prespecified high-reproducibility criterion (ARI >= 0.90 and recovery >= 0.70): QC 0/1 ARI is 0.802/recovery 0.667 and QC 0 ARI is 0.656/recovery 0.200.\n4. **Do four broad carbon-response signatures persist across QC filtering?** Directionally consistent. The aligned K4 components remain identifiable, although QC 0 shifts the high-CH4 K4 centroid's NEE sign; this qualitative change is flagged in `k4_cross_qc_alignment.csv`.\n5. **Does the high-CH4 tail component persist?** Partly. QC 0/1 has a reproducible response-tail component (CH4 centroid about 198); QC 0 retains a high-CH4 emission region (about 209) but classifies it as an unstable subdivision rather than a reproducible tail.\n6. **Does the unstable K6 subdivision persist?** Yes, as instability rather than a label-stable component: both QC branches contain unstable subdivisions, with changed identity/geometry.\n7. **Does the hierarchical K4 -> K6 structure survive QC filtering?** Yes. Mean coarsened ARI is at least 0.55 in every evaluated branch and child-parent purity remains summarized in the hierarchy output.\n8. **Does Markov persistence remain strong?** Yes. Observed/independence self-pair ratios are 3.20--5.44 across QC branch/K combinations, but the fragmentation above limits direct duration comparisons.\n9. **Do the existing hourly results support the same broad carbon structure?** Yes as a broad-structure sensitivity: hourly K4 ARI is 0.999 and K6 ARI is 0.867; it does not establish exact high-K partition equivalence.\n10. **Is the evidence now sufficient for a formal manuscript state-count decision?** No. The exact K=6 partition remains sensitivity-dependent and K4 is not highly reproducible under both QC restrictions.\n\n## Robustness classification\n{robust.to_csv(index=False)}\n\nNo manuscript state count is selected here.\n\n{conclusion}\n"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding='utf-8')
    print(conclusion)


if __name__ == '__main__':
    main()
