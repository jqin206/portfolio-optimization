import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from generator import generate_portfolio
from normalizer import compute_scores, factors
from optimizer import (
    strategies, macro_conditions, strategy_risk_aversion,
    get_blended_utility, enforce_tranches, run_optimization, make_bounds, random_x_init,
    TOTAL_BUDGET, MIN_WEIGHT, MAX_WEIGHT, TRANCHE_SIZE,
)

BASELINE_SEED = 42

NOISE_SEEDS = [21, 22, 23, 24, 25]
NOISE_SCALE = 0.05

INIT_SEEDS = [11, 12, 13, 14, 15]
MAX_WEIGHT_GRID = [0.10, 0.15, 0.20, 0.25]
RISK_AVERSION_MULTIPLIERS = [0.5, 0.75, 1.0, 1.25, 1.5]

metric_cols = [c for cols in factors.values() for c in cols]

def _row(axis, perturbation_value, strat_name, macro_name, startup_id, dollars, converged):
    return {
        'axis': axis,
        'perturbation_value': perturbation_value,
        'strategy': strat_name,
        'regime': macro_name,
        'startup_id': startup_id,
        'dollars_allocated': dollars,
        'converged': converged,
    }

def _summarize(axis, strat_name, macro_name, metrics):
    metrics_df = pd.DataFrame(metrics)
    return {
        'axis': axis,
        'strategy': strat_name,
        'regime': macro_name,
        'mean_abs_delta': metrics_df['mean_abs_delta'].mean(),
        'max_abs_delta': metrics_df['max_abs_delta'].max(),
        'pct_budget_reallocated': metrics_df['pct_budget_reallocated'].mean(),
        'spearman_corr': metrics_df['spearman_corr'].min(),
        'hhi_drift': (metrics_df['hhi_perturbed'] - metrics_df['hhi_baseline']).mean(),
        'n_failed_convergence': int((~metrics_df['converged']).sum()),
        'n_runs': len(metrics_df),
    }

def herfindahl_index(dollars):
    w = dollars / dollars.sum()
    return float(np.sum(w ** 2))

def compare_to_baseline(baseline_dollars, perturbed_dollars):
    abs_delta = np.abs(perturbed_dollars - baseline_dollars)
    corr, _ = spearmanr(baseline_dollars, perturbed_dollars)
    return {
        'mean_abs_delta': abs_delta.mean(),
        'max_abs_delta': abs_delta.max(),
        'pct_budget_reallocated': abs_delta.sum() / (2 * TOTAL_BUDGET),
        'spearman_corr': corr,
        'hhi_baseline': herfindahl_index(baseline_dollars),
        'hhi_perturbed': herfindahl_index(perturbed_dollars),
    }

def jitter_metrics(df, seed, noise_scale):
    rng = np.random.default_rng(seed)
    noisy = df.copy()
    for col in metric_cols:
        noise = rng.normal(loc=1.0, scale=noise_scale, size=len(df))
        noisy[col] = (df[col] * noise).clip(lower=df[col].min(), upper=df[col].max())
    return noisy

def build_noisy_datasets(baseline_df):
    out = {}
    for seed in NOISE_SEEDS:
        noisy_df = jitter_metrics(baseline_df, seed, NOISE_SCALE)
        scores_df, _, semicov_df = compute_scores(noisy_df)
        out[seed] = (scores_df, semicov_df.values)
    return out

def build_alt_inits(num_startups):
    return {seed: random_x_init(num_startups, seed) for seed in INIT_SEEDS}

def sensitivity_to_metric_noise(strat_name, macro_name, strat_w, macro_m, risk_aversion,
                                 bounds, x_init, alt_noisy_datasets, baseline_dollars, ids):
    runs, metrics = [], []
    for seed, (scores_df, Sigma) in alt_noisy_datasets.items():
        utility_vector = get_blended_utility(strat_w, macro_m, scores_df)
        res = run_optimization(Sigma, utility_vector, risk_aversion, bounds, x_init)
        dollars = enforce_tranches(res.x, TOTAL_BUDGET, TRANCHE_SIZE, MAX_WEIGHT * TOTAL_BUDGET)
        runs += [_row('metric_noise', seed, strat_name, macro_name, i, d, res.success) for i, d in zip(ids, dollars)]
        metrics.append(compare_to_baseline(baseline_dollars, dollars) | {'converged': res.success})
    return runs, _summarize('metric_noise', strat_name, macro_name, metrics)

def sensitivity_to_init(strat_name, macro_name, Sigma, utility_vector, risk_aversion,
                         bounds, alt_inits, baseline_dollars, ids):
    runs, metrics = [], []
    for seed, x_init in alt_inits.items():
        res = run_optimization(Sigma, utility_vector, risk_aversion, bounds, x_init)
        dollars = enforce_tranches(res.x, TOTAL_BUDGET, TRANCHE_SIZE, MAX_WEIGHT * TOTAL_BUDGET)
        runs += [_row('init_seed', seed, strat_name, macro_name, i, d, res.success) for i, d in zip(ids, dollars)]
        metrics.append(compare_to_baseline(baseline_dollars, dollars) | {'converged': res.success})
    return runs, _summarize('init_seed', strat_name, macro_name, metrics)

def sensitivity_to_cap(strat_name, macro_name, Sigma, utility_vector, risk_aversion,
                        x_init, num_startups, baseline_dollars, ids):
    runs, metrics = [], []
    for max_weight in MAX_WEIGHT_GRID:
        bounds = make_bounds(num_startups, MIN_WEIGHT, max_weight)
        res = run_optimization(Sigma, utility_vector, risk_aversion, bounds, x_init)
        dollars = enforce_tranches(res.x, TOTAL_BUDGET, TRANCHE_SIZE, max_weight * TOTAL_BUDGET)
        runs += [_row('max_weight', max_weight, strat_name, macro_name, i, d, res.success) for i, d in zip(ids, dollars)]
        metrics.append(compare_to_baseline(baseline_dollars, dollars) | {'converged': res.success})
    return runs, _summarize('max_weight', strat_name, macro_name, metrics)

def sensitivity_to_risk_aversion(strat_name, macro_name, Sigma, utility_vector,
                                  bounds, x_init, baseline_dollars, ids):
    runs, metrics = [], []
    for multiplier in RISK_AVERSION_MULTIPLIERS:
        res = run_optimization(Sigma, utility_vector, strategy_risk_aversion[strat_name] * multiplier, bounds, x_init)
        dollars = enforce_tranches(res.x, TOTAL_BUDGET, TRANCHE_SIZE, MAX_WEIGHT * TOTAL_BUDGET)
        runs += [_row('risk_aversion', multiplier, strat_name, macro_name, i, d, res.success) for i, d in zip(ids, dollars)]
        metrics.append(compare_to_baseline(baseline_dollars, dollars) | {'converged': res.success})
    return runs, _summarize('risk_aversion', strat_name, macro_name, metrics)

def run_validation():
    baseline_df = generate_portfolio(seed=BASELINE_SEED)
    num_startups = len(baseline_df)
    baseline_scores, _, baseline_Sigma = compute_scores(baseline_df)
    ids = baseline_scores['id'].values

    bounds = make_bounds(num_startups, MIN_WEIGHT, MAX_WEIGHT)
    x_init = np.ones(num_startups) / num_startups

    alt_noisy_datasets = build_noisy_datasets(baseline_df)

    alt_inits = build_alt_inits(num_startups)

    all_runs, all_summaries = [], []

    for strat_name, strat_w in strategies.items():
        for macro_name, macro_m in macro_conditions.items():
            risk_aversion = strategy_risk_aversion[strat_name]
            utility_vector = get_blended_utility(strat_w, macro_m, baseline_scores)
            baseline_res = run_optimization(baseline_Sigma, utility_vector, risk_aversion, bounds, x_init)
            baseline_dollars = enforce_tranches(baseline_res.x, TOTAL_BUDGET, TRANCHE_SIZE, MAX_WEIGHT * TOTAL_BUDGET)

            for runs, summary in [
                sensitivity_to_metric_noise(strat_name, macro_name, strat_w, macro_m, risk_aversion,
                                 bounds, x_init, alt_noisy_datasets, baseline_dollars, ids),
                sensitivity_to_init(strat_name, macro_name, baseline_Sigma, utility_vector, risk_aversion,
                                    bounds, alt_inits, baseline_dollars, ids),
                sensitivity_to_cap(strat_name, macro_name, baseline_Sigma, utility_vector, risk_aversion,
                                   x_init, num_startups, baseline_dollars, ids),
                sensitivity_to_risk_aversion(strat_name, macro_name, baseline_Sigma, utility_vector,
                             bounds, x_init, baseline_dollars, ids),
            ]:
                all_runs += runs
                all_summaries.append(summary)

    pd.DataFrame(all_runs).to_csv('portfolio/validation_runs.csv', index=False)
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv('portfolio/validation_summary.csv', index=False)

    flagged = summary_df[(summary_df['spearman_corr'] < 0.8) | (summary_df['pct_budget_reallocated'] > 0.15)]
    if not flagged.empty:
        print("High-sensitivity combos:\n", flagged)

if __name__ == "__main__":
    run_validation()
