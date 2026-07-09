import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_scores = pd.read_csv('portfolio/scores.csv')

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
plt.savefig("portfolio/startup_raw_metrics_heatmap.png", dpi=300)
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
plt.savefig("portfolio/startup_quadrant_bubble_chart.png", dpi=300)
plt.close()


df_raw = pd.read_csv('portfolio/mock_portfolio.csv')
cov_matrix = pd.read_csv('portfolio/semicovariance.csv', index_col=0).values
df_sim = pd.read_csv('simulation.csv')

modifiers = {
    'Bull Market':       {'growth_mult': 1.5, 'risk_mult': 0.6},
    'Recession':         {'growth_mult': 0.5, 'risk_mult': 1.6},
    'Stagflation':       {'growth_mult': 0.8, 'risk_mult': 1.0},
    'Neutral Market':    {'growth_mult': 1.0, 'risk_mult': 1.0}
}

strategy_cols = [c for c in df_sim.columns if c != 'id']
portfolio_data = []

for col in strategy_cols:
    allocations = df_sim[col].values
    weights = allocations / np.sum(allocations)
    
    if '_in_bull' in col:         color_cat = 'Bull Market'
    elif '_in_recession' in col:  color_cat = 'Recession'
    elif '_in_stagflation' in col: color_cat = 'Stagflation'
    elif '_in_neutral' in col:      color_cat = 'Neutral Market'
        
    mod = modifiers[color_cat]
    
    base_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    p_risk = base_risk * mod['risk_mult']
    
    raw_growth = df_raw['yoy_revenue_growth'].values
    scaled_growth = raw_growth * mod['growth_mult']
    p_return_percentage = np.dot(weights, scaled_growth)
    
    strat_name = col.split('_in_')[0].split('_expected')[0].upper()
    
    if 'GROWTH' in strat_name:       shape_cat = 'Growth'
    elif 'RISK' in strat_name:        shape_cat = 'Risk'
    elif 'EFFICIENCY' in strat_name: shape_cat = 'Efficiency'
    elif 'STRATEGIC' in strat_name:  shape_cat = 'Strategic'
    else:                             shape_cat = 'Baseline'
    
    portfolio_data.append({
        'Strategy': strat_name,
        'Risk': p_risk,
        'Return_Pct': p_return_percentage,
        'Shape_Cat': shape_cat,
        'Color_Cat': color_cat
    })

df_plot_all = pd.DataFrame(portfolio_data)

markers = {
    'Baseline': 'o',
    'Growth': '^',
    'Risk': 'v',
    'Efficiency': 's',
    'Strategic': 'D',
}
colors = {
    'Bull Market': 'gold', 
    'Recession': 'green', 
    'Stagflation': 'blue', 
    'Neutral Market': 'purple'
}

fig, ax = plt.subplots(figsize=(11, 7.5))
np.random.seed(42)
x_jitter_range = 0.003
y_jitter_range = 0.015

for regime_name, regime_color in colors.items():
    
    df_plot = df_plot_all[df_plot_all['Color_Cat'] == regime_name]
    
    if df_plot.empty:
        continue
        
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    np.random.seed(42)
    
    for (shape_cat, color_cat), group in df_plot.groupby(['Shape_Cat', 'Color_Cat']):
        x_noise = np.random.uniform(-x_jitter_range, x_jitter_range, size=len(group))
        y_noise = np.random.uniform(-y_jitter_range, y_jitter_range, size=len(group))
        
        ax.scatter(
            group['Risk'] + x_noise,       
            group['Return_Pct'] + y_noise, 
            marker=markers[shape_cat], 
            c=regime_color, 
            s=150, 
            edgecolors='k',
            alpha=0.85, 
            label='_nolegend_'
        )
    
    # Render Plain Text Legend Headers
    ax.scatter([], [], color='none', label=f"STRATEGIES")
    for shape_name, marker_shape in markers.items():
        if shape_name in df_plot['Shape_Cat'].values:
            ax.scatter(
                [], [], 
                marker=marker_shape, 
                color=regime_color, 
                edgecolors='k', 
                s=100, 
                label=shape_name
            )
    
    # Handle Legend Positioning
    if regime_name == 'Recession':
        target_loc = "upper right"
    else:
        target_loc = "upper left"
        
    leg = ax.legend(loc=target_loc, frameon=True, edgecolor="lightgray", fontsize=9, labelspacing=0.5)

    if leg.get_texts():
        leg.get_texts()[0].set_weight('bold')
    
    # Labels and Titles
    ax.set_xlabel("Downside Portfolio Risk")
    ax.set_ylabel("Portfolio Return (Weighted Average YoY Revenue Growth)")
    ax.set_title(f"Portfolio Risk vs. Return ({regime_name})", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save Outputs
    file_suffix = regime_name.lower().replace(' ', '_')
    plt.savefig(f"regime_graphs/risk_vs_return_{file_suffix}.png", dpi=300)
    plt.close()

fig, ax = plt.subplots(figsize=(11, 7.5))

for (shape_cat, color_cat), group in df_plot_all.groupby(['Shape_Cat', 'Color_Cat']):
    x_noise = np.random.uniform(-x_jitter_range, x_jitter_range, size=len(group))
    y_noise = np.random.uniform(-y_jitter_range, y_jitter_range, size=len(group))
    
    ax.scatter(
        group['Risk'] + x_noise,       
        group['Return_Pct'] + y_noise, 
        marker=markers[shape_cat], 
        c=colors[color_cat], 
        s=140, 
        edgecolors='k',
        alpha=0.85, 
        label='_nolegend_'
    )

ax.scatter([], [], color='none', label=r'$\bf{STRATEGIES}$')
for strategy, marker_shape in markers.items():
    ax.scatter(
        [], [], 
        marker=marker_shape, 
        c='gray',
        edgecolors='k', 
        s=120, 
        label=strategy
    )

ax.scatter([], [], color='none', label=r'$\bf{MARKET\ CONDITIONS}$')
for regime, regime_color in colors.items():
    ax.scatter(
        [], [], 
        marker='o',
        c=regime_color, 
        edgecolors='k', 
        s=120, 
        label=regime
    )

ax.legend(
    loc="upper right", 
    frameon=True, 
    edgecolor="lightgray", 
    fontsize=10, 
    labelspacing=0.4
)

ax.set_xlabel("Downside Portfolio Risk", fontweight='bold')
ax.set_ylabel("Portfolio Return (Weighted Average YoY Revenue Growth)", fontweight='bold')
ax.set_title("Portfolio Risk vs. Return", fontsize=13, fontweight='bold', pad=15)
ax.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig("regime_graphs/risk_vs_return.png", dpi=300)
plt.close()

regimes = {
    'neutral': 'Neutral Market',
    'bull': 'Bull Market',
    'recession': 'Recession',
    'stagflation': 'Stagflation',
}

for suffix, display_name in regimes.items():
    regime_cols = [c for c in df_sim.columns if f'_in_{suffix}' in c]
    heatmap_data = round(df_sim[regime_cols] / 10_000_000.0, 3)
    y_labels = [c.replace(f'_in_{suffix}', '').strip() for c in heatmap_data.columns]
    heatmap_data.columns = y_labels

    plt.figure(figsize=(18, 5))
    sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", cmap="YlGnBu",
                cbar_kws={'label': 'Allocated Capital ($)'}, square=True)

    plt.title(f"Portfolio Capital Allocation in {display_name}", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Startup ID", fontsize=12, labelpad=10)
    plt.ylabel("Strategies", fontsize=12)

    plt.tight_layout()
    plt.savefig(f"regime_graphs/portfolio_allocation_{suffix}_heatmap.png", dpi=300)
    plt.close()
