# Startup Portfolio Capital Allocation

This pipeline computes growth, risk, capital efficiency, and strategic importance scores from raw startup data and utilizes these scores to produce the optimal capital allocation for the given portfolio.

## Allocation constraints
Total capital pool: $10 million
Minimum allocation per startup: $50,000
Maximum allocation per startup: $1.5 million (15% of capital pool)

## Structure

### 1. Data generator (`generator.py`)
- Produces a mock portfolio of 20 startups with the following data.
- Output stored in `mock_portfolio.csv`.

### 2. Data normalizer (`normalizer.py)
- Normalizes portfolio data to z-scores.
- Calculates growth, risk, capital efficiency, and strategic importance scores from raw data using Principal Component Analysis (PCA).
- Computes the portfolio's semi-covariance matrix.
- Outputs stored in `pca_factor_weights.csv`, `scores.csv`, and `semicovariance.csv`.

### 3. Portfolio optimizer (`optimizer.py`)
- Recommends capital allocation for 6 different strategies in 3 market conditions using sequential least squares programming.
- Output stored in `simulation.csv`.

### 4. Data visualizer (`visualizer.py`)
- Creates heatmaps for the startup profiles (`startup_raw_metrics_heatmap.png`) and different allocation scenarios (`portfolio_allocation_heatmap.png`).
- Visualizes the risk vs. return tradeoff for each allocation strategy in different market conditions (`startup_quadrant_bubble_chart.png`).