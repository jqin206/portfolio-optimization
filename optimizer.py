import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.optimize import minimize

strategies = {
    'aggressive': {'growth': 0.60, 'risk': 0.10, 'capital_efficiency': 0.10, 'strategic_importance': 0.20},
    'safety': {'growth': 0.10, 'risk': 0.40, 'capital_efficiency': 0.40, 'strategic_importance': 0.10},
    'balanced': {'growth': 0.30, 'risk': 0.20, 'capital_efficiency': 0.20, 'strategic_importance': 0.30},
}

macro_conditions = {
    'bull': {'growth': 1.5, 'risk': 0.6, 'capital_efficiency': 0.8, 'strategic_importance': 1.1},
    'recession': {'growth': 0.5, 'risk': 1.6, 'capital_efficiency': 1.4, 'strategic_importance': 0.5},
    'stagflation': {'growth': 0.8, 'risk': 1.0, 'capital_efficiency': 1.3, 'strategic_importance': 0.9},
}

df = pd.read_csv('raw_scores.csv')
num_startups = len(df)

TOTAL_BUDGET = 10_000_000
MIN_WEIGHT = 0.01
MAX_WEIGHT = 0.2

RISK_AVERSION = 2.0

Sigma = pd.read_csv("portfolio_semi_covariance.csv", index_col=0).values

constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1.0}
bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(num_startups)]

x_init = np.ones(num_startups) * (1.0 / num_startups)

master_matrix = pd.DataFrame({'id': df['id']})

for (strat_name, strat_w), (macro_name, macro_m) in itertools.product( strategies.items(), macro_conditions.items()):
    combined_raw = { factor: strat_w[factor] * macro_m[factor] for factor in strat_w }

    total_w = sum(combined_raw.values())
    blended_weights = { p: w / total_w for p, w in combined_raw.items() }

    utility_vector = (
        (df['growth_score'] * blended_weights['growth'])
        + (df['risk_score'] * blended_weights['risk'])
        + (df['capital_efficiency_score'] * blended_weights['capital_efficiency'])
        + (df['strategic_importance_score'] * blended_weights['strategic_importance'])
    ).values

    def objective(x):
        portfolio_variance = np.dot(x.T, np.dot(Sigma, x))
        portfolio_return = np.dot(np.log(utility_vector + 1), x)
        return portfolio_variance - RISK_AVERSION * portfolio_return

    res = minimize(
        objective,
        x_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if res.success:
        column_label = f"{strat_name}_in_{macro_name}"
        master_matrix[column_label] = (res.x * TOTAL_BUDGET).round(0)
    else:
        print(f"Warning: Optimization failed for {strat_name}_in_{macro_name}")

master_matrix.to_csv("quadratic_simulation_cross_matrix.csv", index=False)