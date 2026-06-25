import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

df = pd.read_csv('portfolio_mock.csv')
num_startups = len(df)

factors = {
    'growth': ['yoy_revenue_growth', 'nrr', 'customer_growth_rate', 'tam_bn', 'pmf_indicator'],
    'risk': ['runway_months', 'customer_churn', 'revenue_concentration_risk', 'market_regulatory_risk', 'execution_risk'],
    'capital_efficiency': ['monthly_burn_rate_k', 'ltv_cac', 'cac_payback_months', 'revenue_per_employee', 'gross_margin'],
    'strategic_importance': ['strategic_alignment', 'portfolio_synergies', 'market_positioning', 'future_fundraising_potential', 'competitive_differentiation']
}

print(df[factors['growth']].corr())
print(df[factors['risk']].corr())
print(df[factors['capital_efficiency']].corr())
print(df[factors['strategic_importance']].corr())

# Compute normalized growth, risk, capital efficiency, and strategic importance scores

weights_records = []
final_scores_df = pd.DataFrame({'id': df['id']})

for factor, cols in factors.items():
    raw_data = df[cols].copy()
    if factor == 'risk':
        hazard_metrics = ['customer_churn', 'revenue_concentration_risk', 'market_regulatory_risk', 'execution_risk']
        for col in hazard_metrics:
            raw_data[col] = raw_data[col].max() - raw_data[col]
    elif factor == 'capital_efficiency':
        raw_data['monthly_burn_rate_k'] = (raw_data['monthly_burn_rate_k'].max() - raw_data['monthly_burn_rate_k'])
        raw_data['cac_payback_months'] = (raw_data['cac_payback_months'].max() - raw_data['cac_payback_months'])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)

    pca = PCA(n_components=1)
    composite_score = pca.fit_transform(scaled_data)
    weights = pca.components_[0]

    if np.mean(weights) < 0:
        composite_score = -composite_score
        weights = -weights

    for col_name, weight_value in zip(cols, weights):
        weights_records.append(
            {
                'factor': factor,
                'metric': col_name,
                'pca_weight': round(weight_value, 4)
            }
        )

    min_max_scaler = MinMaxScaler(feature_range=(0, 10))
    scaled_scores = min_max_scaler.fit_transform(composite_score)

    final_scores_df[f'{factor}_score'] = scaled_scores.flatten().round(2)

print(final_scores_df.head())
final_scores_df.to_csv('raw_scores.csv', index=False)

weights_df = pd.DataFrame(weights_records)
print(weights_df.head())
weights_df.to_csv("pca_factor_weights.csv", index=False)

# Create semi-covariance matrix
all_metric_cols = []
for factor_name, cols_list in factors.items():
    all_metric_cols.extend(cols_list)

cov_raw_data = df[all_metric_cols].copy()

for col in factors['risk']:
    cov_raw_data[col] = cov_raw_data[col].max() - cov_raw_data[col]
cov_raw_data['monthly_burn_rate_k'] = cov_raw_data['monthly_burn_rate_k'].max() - cov_raw_data['monthly_burn_rate_k']
cov_raw_data['cac_payback_months'] = cov_raw_data['cac_payback_months'].max() - cov_raw_data['cac_payback_months']

cov_scaler = StandardScaler()
scaled_ops_data = cov_scaler.fit_transform(cov_raw_data)

feature_means = np.mean(scaled_ops_data, axis=0)
deviations = scaled_ops_data - feature_means
downside_deviations = np.minimum(deviations, 0)

semi_cov_matrix = np.dot(downside_deviations, downside_deviations.T) / (num_startups - 1)
semi_cov_matrix += np.eye(num_startups) * 0.005

semi_covariance_df = pd.DataFrame(
    semi_cov_matrix, 
    index=df['id'],   
    columns=df['id']  
)
print(semi_covariance_df.head())
semi_covariance_df.to_csv("portfolio_semi_covariance.csv", index=True)
