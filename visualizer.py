import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_scores = pd.read_csv('scores.csv')

metrics_cols = ['growth_score', 'risk_score', 'capital_efficiency_score', 'strategic_importance_score']
display_cols = ['Growth Score', 'Risk Score', 'Capital Efficiency', 'Strategic Importance']

fig1, ax1 = plt.subplots(figsize=(10, 11))

heatmap_data = df_scores.set_index('id')[metrics_cols]
heatmap_data.columns = display_cols

sns.heatmap(
    heatmap_data, 
    annot=True, 
    fmt=".1f", 
    cmap="Blues", 
    ax=ax1, 
    cbar_kws={'label': 'Raw Attribute Score (Scale: $1 - 10$)'}
)

ax1.set_title("Startup Profiles", pad=20, fontsize=14, weight='bold')
ax1.set_ylabel("Startup ID", fontsize=12)
ax1.set_xlabel("Operational Metrics", fontsize=12)
plt.xticks(rotation=15, ha='right', rotation_mode='anchor')

plt.tight_layout()
plt.savefig("startup_raw_metrics_heatmap.png", dpi=300)
plt.close()

fig2, ax2 = plt.subplots(figsize=(11, 8))

scatter = ax2.scatter(
    df_scores['growth_score'],
    df_scores['risk_score'],
    s=(df_scores['capital_efficiency_score'] + 1.5) * 65,
    c=df_scores['strategic_importance_score'],
    cmap='viridis',
    alpha=0.80,
    edgecolors='w',
    linewidth=1.2
)

for i, row in df_scores.iterrows():
    ax2.text(
        row['growth_score'],
        row['risk_score'] + 0.2,
        f"{int(row['id'])}",
        ha='center',
        va='bottom',
        fontsize=10,
        weight='bold'
    )

ax2.set_title("Growth vs. Risk Trajectories", pad=20, fontsize=14, weight='bold')
ax2.set_xlabel("Growth Score", fontsize=12)
ax2.set_ylabel("Risk Score", fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.4)

cbar = fig2.colorbar(scatter, ax=ax2)
cbar.set_label('Strategic Importance Score', fontsize=11)

sizes = [2.0, 5.0, 8.0, 10.0]
legend_markers = [ax2.scatter([], [], s=sz * 65, c='gray', alpha=0.5, edgecolors='w') for sz in sizes]
ax2.legend(
    legend_markers, 
    [f"{sz:.1f}" for sz in sizes], 
    loc="upper left",
    title="Capital Efficiency",
    ncol=5,
    columnspacing=2,
    borderpad=1.0
)

plt.tight_layout()
plt.savefig("startup_quadrant_bubble_chart.png", dpi=300)
plt.close()

expected_returns = (df_scores['growth_score'] * 0.4 + df_scores['capital_efficiency_score'] * 0.3) / 10.0 * 0.35
cov_df = pd.read_csv('semicovariance.csv', index_col=0)
cov_matrix = cov_df.to_numpy()

df_sim = pd.read_csv('simulation.csv')
strategy_cols = [c for c in df_sim.columns if c != 'id']

portfolio_data = []

for strategy in strategy_cols:
    allocations = df_sim[strategy].values
    weights = allocations / np.sum(allocations)
    
    p_return = np.dot(weights, expected_returns)
    p_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    portfolio_data.append({
        'Strategy': strategy.replace('_expected', '').upper(),
        'Risk': p_risk,
        'Return': p_return
    })

df_plot = pd.DataFrame(portfolio_data)
fig, ax = plt.subplots(figsize=(10, 6.5))

def get_shape_category(strat):
    if 'GROWTH_RISK' in strat: return 'Growth Risk'
    if 'RISK_EFF' in strat:    return 'Risk Efficiency'
    if 'BASELINE' in strat:    return 'Baseline'
    if 'GROWTH' in strat:      return 'Growth'
    if 'STRATEGIC' in strat:   return 'Strategic'
    if 'EFFICIENCY' in strat:   return 'Efficiency'
    return 'Other'

def get_color_category(strat):
    if 'BULL' in strat: return 'Bull Market'
    if 'RECESSION' in strat: return 'Recession'
    if 'STAGFLATION' in strat: return 'Stagflation'
    return 'Expected Baseline'

df_plot['Shape_Cat'] = df_plot['Strategy'].apply(get_shape_category)
df_plot['Color_Cat'] = df_plot['Strategy'].apply(get_color_category)

markers = {
    'Baseline': 'o',
    'Growth': '^',
    'Growth Risk': 'X',
    'Risk Efficiency': 'v',
    'Efficiency': 's',
    'Strategic': 'D',
}
colors = {'Bull Market': 'gold', 'Recession': 'green', 'Stagflation': 'blue', 'Expected Baseline': 'purple'}

for (shape_cat, color_cat), group in df_plot.groupby(['Shape_Cat', 'Color_Cat']):
    ax.scatter(
        group['Risk'], 
        group['Return'], 
        marker=markers[shape_cat], 
        c=colors[color_cat], 
        s=150, 
        edgecolors='k',
        label='_nolegend_'
    )

ax.scatter([], [], color='none', label=r"$\bf{STRATEGIES}$")
for shape_name, marker_shape in markers.items():
    ax.scatter(
        [], [], 
        marker=marker_shape, 
        color='gray',
        edgecolors='k', 
        s=100, 
        label=shape_name
    )

ax.scatter([], [], color='none', label="") 

ax.scatter([], [], color='none', label=r"$\bf{MARKET\ CONDITIONS}$")
for regime, color_name in colors.items():
    if regime != 'Expected Baseline':
        ax.scatter([], [], marker='o', color=color_name, edgecolors='k', s=100, label=regime)

ax.legend(
    loc="lower right", 
    frameon=True, 
    edgecolor="lightgray", 
    fontsize=9,
    labelspacing=0.5
)

plt.tight_layout()
plt.savefig("risk_return_tradeoff.png", dpi=300)
plt.close()

expected_cols = [c for c in df_sim.columns if 'expected' in c]
heatmap_data = df_sim[expected_cols]

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Allocated Capital ($)'})
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

plt.title("Expected Portfolio Capital Allocation By Startup")
plt.ylabel("Startup ID")
plt.xlabel("Strategies")
plt.tight_layout()
plt.savefig("portfolio_allocation_heatmap.png", dpi=300)
plt.close()