import numpy as np
import pandas as pd
from scipy.optimize import minimize

strategies = {
    'growth': {'growth': 0.7, 'risk': 0.1, 'capital_efficiency': 0.1, 'strategic_importance': 0.1},
    'risk': {'growth': 0.1, 'risk': 0.7, 'capital_efficiency': 0.1, 'strategic_importance': 0.1},
    'efficiency': {'growth': 0.1, 'risk': 0.1, 'capital_efficiency': 0.7, 'strategic_importance': 0.1},
    'strategic': {'growth': 0.1, 'risk': 0.1, 'capital_efficiency': 0.1, 'strategic_importance': 0.7},
    'baseline': {'growth': 0.25, 'risk': 0.25, 'capital_efficiency': 0.25, 'strategic_importance': 0.25},
}

macro_conditions = {
    'bull': {'growth': 1.5, 'risk': 0.6, 'capital_efficiency': 0.8, 'strategic_importance': 1.1},
    'recession': {'growth': 0.5, 'risk': 1.6, 'capital_efficiency': 1.4, 'strategic_importance': 0.5},
    'stagflation': {'growth': 0.8, 'risk': 1.0, 'capital_efficiency': 1.3, 'strategic_importance': 0.9},
}

macro_probabilities = {
    'bull': 0.50,
    'recession': 0.30,
    'stagflation': 0.20
}

df = pd.read_csv('scores.csv')
num_startups = len(df)

TOTAL_BUDGET = 10_000_000
MIN_WEIGHT = 0.005
MAX_WEIGHT = 0.15

TRANCHE_SIZE = 50_000
MAX_CHECK_SIZE = MAX_WEIGHT * TOTAL_BUDGET

strategy_risk_aversion = {
    'growth': 0.5,
    'risk': 5.0,
    'efficiency': 1.5,
    'strategic': 3.0,
    'baseline': 2.5
}

Sigma = pd.read_csv("semicovariance.csv", index_col=0).values

constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
bounds = [(MIN_WEIGHT, MAX_WEIGHT) for _ in range(num_startups)]

x_init = np.ones(num_startups) * (1.0 / num_startups)

master_matrix = pd.DataFrame({'id': df['id']})

def portfolio_objective(x, Sigma, utility_vector, risk_aversion):
    portfolio_variance = np.dot(x.T, np.dot(Sigma, x))
    portfolio_return = np.dot(np.log(utility_vector + 1), x)
    return portfolio_variance - risk_aversion * portfolio_return

def get_blended_utility(strat_w, macro_m, df_source):
    combined_raw = {factor: strat_w[factor] * macro_m[factor] for factor in strat_w}
    total_w = sum(combined_raw.values())
    blended_weights = {p: w / total_w for p, w in combined_raw.items()}
    
    return (
        (df_source['growth_score'] * blended_weights['growth'])
        + (df_source['risk_score'] * blended_weights['risk'])
        + (df_source['capital_efficiency_score'] * blended_weights['capital_efficiency'])
        + (df_source['strategic_importance_score'] * blended_weights['strategic_importance'])
    ).values

def enforce_tranches(raw_weights, total_budget, tranche_size, max_check):
    raw_dollars = raw_weights * total_budget
    
    tranche_allocations = np.floor(raw_dollars / tranche_size) * tranche_size
    
    leftover_budget = total_budget - np.sum(tranche_allocations)
    
    rounding_losses = raw_dollars - tranche_allocations
    priority_indices = np.argsort(-rounding_losses)
    
    for idx in priority_indices:
        if leftover_budget <= 0:
            break
        if tranche_allocations[idx] + tranche_size <= max_check:
            tranche_allocations[idx] += tranche_size
            leftover_budget -= tranche_size
            
    return tranche_allocations

for strat_name, strat_w in strategies.items():
    RISK_AVERSION = strategy_risk_aversion[strat_name]
    expected_utility_vector = np.zeros(num_startups)
    
    for macro_name, macro_m in macro_conditions.items():
        utility_vector = get_blended_utility(strat_w, macro_m, df)
        
        prob = macro_probabilities[macro_name]
        expected_utility_vector += prob * utility_vector
        
        res_ind = minimize(portfolio_objective, x_init, args=(Sigma, utility_vector, RISK_AVERSION),
                           method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res_ind.success:
            clean_dollars = enforce_tranches(res_ind.x, TOTAL_BUDGET, TRANCHE_SIZE, MAX_CHECK_SIZE)
            master_matrix[f"{strat_name}_in_{macro_name}"] = clean_dollars

    res_exp = minimize(portfolio_objective, x_init, args=(Sigma, expected_utility_vector, RISK_AVERSION),
                       method='SLSQP', bounds=bounds, constraints=constraints)
    
    if res_exp.success:
        clean_expected_dollars = enforce_tranches(res_exp.x, TOTAL_BUDGET, TRANCHE_SIZE, MAX_CHECK_SIZE)
        master_matrix[f'{strat_name}_expected'] = clean_expected_dollars
    else:
        print(f'Warning: Expected allocation calculation failed for {strat_name}')

master_matrix.to_csv('simulation.csv', index=False)