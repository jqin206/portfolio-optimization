# Startup Portfolio Capital Allocation

This pipeline computes growth, risk, capital efficiency, and strategic importance scores from raw startup data and utilizes these scores to produce the optimal capital allocation for the given portfolio.

## Allocation constraints
**Total capital pool:** $10 million

**Minimum allocation per startup:** $50,000

**Maximum allocation per startup:** $1.5 million (15% of capital pool)


## Methodology

1. **Startup data generation**: provide raw startup business metrics to input into the model.

2. **Composite scoring via PCA**: compute growth, risk, capital efficiency, and strategic importance scores for each startup based on its raw metrics.

3. **Semi-covariance matrix**: assess how startups' downside risk move together so that the optimizer avoids overconcentrating capital in startups that lose value at the same time.

4. **Strategy and market condition definition**: define 5 allocation strategies and 4 macroeconomic scenarios to be tested by the optimizer.

5. **Optimization**: compute the optimal portfolio allocation for each strategy in each market condition by maximizing return and minimizing risk based on composite scores, semi-covariance matrix, risk-aversion, constraints, and initial allocation guess.

6. **Final allocation**: recommends proportion of capital that should be allocated to each startup in the portfolio for each strategy and market condition combination in both CSV format and as a heatmap.

7. **Validation**: analyze the model's sensitivity to assumptions, including generated startup metrics, initial allocation guesses, allocation constraints, and risk-aversion values.

### Model Diagram
```mermaid
flowchart TD
    A["Raw startup metrics"] --> B["Composite scores"]
    A --> C["Semi-covariance matrix"]

    B --> E["Optimizer"]
    C --> E
    D1["5 allocation strategies"] --> E
    D2["4 market conditions"] --> E

    E --> F["Optimal capital allocation<br/>per strategy and market condition combination"]
    F --> G["Validation"]
```

## Structure

### 1. Data generator (`model/generator.py`)
- Produces a mock portfolio of 20 startups with the following data.
- Output stored in `portfolio/mock_portfolio.csv`.

| Field | Description |
| --- | --- |
| `id` | Unique identifier for each startup |
| `stage` | Current stage of the startup (Seed, Series A, Series B) |
| `yoy_revenue_growth` | Year-over-year revenue growth |
| `nrr` | Net revenue retention |
| `customer_growth` | Customer growth rate |
| `tam_bn` | Total addressable market (billions) |
| `pmf_indicator` | Product/market fit indicator (1-10) |
| `runway` | Time (months) until startup runs out of cash |
| `customer_churn` | Rate at which customers stop using the product |
| `revenue_concentration_risk` | Risk from a disproportionate amount of sales coming from a single consumer |
| `market_regulatory_risk` | Risk from changing regulations or policies (1-10) |
| `execution_risk` | Risk that startup will fail to implement its intended plans (1-10) |
| `monthly_burn_rate_k` | Net burn rate per month (thousands of dollars) |
| `ltv_cac` | LTV to CAC ratio |
| `payback_months` | CAC payback period (months) |
| `rev_per_emp` | Revenue per employee |
| `gross_margin` | Percentage of revenue remaining after subtracting COGS |
| `strat_alignment` | Alignment with portfolio themes and long-term vision (1-10) |
| `strat_synergies` | Synergies with other portfolio companies (1-10) |
| `strat_positioning` | Strategic market positioning |
| `future_fundraising` | Potential for future fundraising or partnership opportunities |
| `competitive_diff` | Competitive differentiation |


### 2. Data normalizer (`model/normalizer.py`)
- Normalizes portfolio data to z-scores.
- Calculates growth, risk, capital efficiency, and strategic importance composite scores (1-10) from raw data using Principal Component Analysis (PCA).
- Computes the portfolio's semi-covariance matrix.
- Outputs stored in `portfolio/pca_factor_weights.csv`, `portfolio/scores.csv`, and `portfolio/semicovariance.csv`.

| Score | Description | Fields |
| --- | --- | --- |
| Growth | Measures a startup's revenue traction and expansion | `yoy_revenue_growth`, `nrr`, `customer_growth`, `tam_bn`, `pmf_indicator` |
| Risk | Measures probability of a startup failing | `runway`, `customer_churn`, `revenue_concentration_risk`, `market_regulatory_risk`, `execution_risk` |
| Capital efficiency | Measures how effectively a startup converts its cash flows into enterprise value | `monthly_burn_rate_k`, `ltv_cac`, `payback_months`, `rev_per_emp`, `gross_margin` |
| Strategic importance | Measures a startup's systemic value to larger STEALTH portfolio | `strat_alignment`, `strat_synergies`, `strat_positioning`, `future_fundraising`, `competitive_diff` |

### 3. Portfolio optimizer (`model/optimizer.py`)
- Recommends capital allocation for 5 different strategies in 4 market conditions using sequential least squares programming.
- Output stored in `simulation.csv`.

| Strategy | Description | 
| --- | --- | 
| `growth` | Growth focused: concentrates 70% of weight into rapidly expanding startups. |
| `risk` | Risk-minimizing: directs 70% of focus toward avoiding vulnerable startups. |
| `efficiency` | Capital efficiency focused: fund startups that can survive on their own operational revenue. |
| `strategic` | Strategic importance focused: prioritize startups building core infrastructure that support a broader ecosystem. |
| `baseline` | Balanced: equal weights across all composite metrics. |


### 4. Data visualizer (`model/visualizer.py`)
- Creates heatmaps for the startup composite scores (`portfolio/startup_raw_metrics_heatmap.png`) and different allocation scenarios (`portfolio_allocation_heatmap.png`).
- Create a bubble chart to visualize the startup composite scores (`portfolio/startup_quadrant_bubble_chart.png`).
- Visualizes the risk vs. return tradeoff for each allocation strategy in different market conditions (`risk_vs_return.png`).

## Quick Start
### 1. Create and activate a Python virtual environment.
```bash
# MacOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
# Command Prompt
.\.venv\Scripts\activate.bat
```

### 2. Install dependencies in the virtual environment.
```bash
pip install -r requirements.txt
```

### 3. Generate mock portfolio.
```bash
python model/generator.py
```

### 4. Construct composite metric scores and semicovariance matrix.
```bash
python model/normalizer.py
```

### 5. Run the optimizer.
```bash
python model/optimizer.py
```

### 6. Visualize results.
```bash
python model/visualizer.py
```