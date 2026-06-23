import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# Cleans and normalizes input data
# Compute growth, risk, capital efficiency, and strategic importance scores

df = pd.read_csv('portfolio_mock.csv')

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

final_scores_df = pd.DataFrame({'id': df['id']})

for factor, cols in factors.items():
    raw_data = df[cols].copy()
    if factor == 'risk':
        for col in cols:
            raw_data[col] = raw_data[col].max() - raw_data[col]
    elif factor == 'capital_efficiency':
        raw_data['monthly_burn_rate_k'] = (raw_data['monthly_burn_rate_k'].max() - raw_data['monthly_burn_rate_k'])
        raw_data['cac_payback_months'] = (raw_data['cac_payback_months'].max() - raw_data['cac_payback_months'])
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data)

    pca = PCA(n_components=1)
    composite_score = pca.fit_transform(scaled_data)
    # weights = pca.components_[0]

    min_max_scaler = MinMaxScaler(feature_range=(0, 10))
    scaled_scores = min_max_scaler.fit_transform(composite_score)

    final_scores_df[f'{factor}_score'] = scaled_scores.flatten().round(2)

final_scores_df.to_csv('raw_scores.csv', index=False)
print(final_scores_df.head())