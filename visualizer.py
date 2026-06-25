import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df_raw = pd.read_csv('raw_scores.csv')

metrics_cols = ['growth_score', 'risk_score', 'capital_efficiency_score', 'strategic_importance_score']
display_cols = ['Growth Score', 'Risk Score', 'Capital Efficiency', 'Strategic Importance']

fig1, ax1 = plt.subplots(figsize=(10, 11))

heatmap_data = df_raw.set_index('id')[metrics_cols]
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
    df_raw['growth_score'],
    df_raw['risk_score'],
    s=(df_raw['capital_efficiency_score'] + 1.5) * 65,
    c=df_raw['strategic_importance_score'],
    cmap='viridis',
    alpha=0.80,
    edgecolors='w',
    linewidth=1.2
)

for i, row in df_raw.iterrows():
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


df = pd.read_csv("portfolio_simulation.csv", index_col='id')

expected_cols = [c for c in df.columns if 'expected' in c]
heatmap_data = df[expected_cols]

plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Allocated Capital ($)'})
plt.xticks(rotation=45, ha='right', rotation_mode='anchor')

plt.title("Expected Portfolio Capital Allocation By Startup")
plt.ylabel("Startup ID")
plt.xlabel("Strategies")
plt.tight_layout()
plt.savefig("portfolio_allocation_heatmap.png", dpi=300)
plt.close()