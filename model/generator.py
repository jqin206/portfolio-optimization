import numpy as np
import pandas as pd

def generate_portfolio(seed=42, num_startups=20) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    stages =  ["Seed"] * (num_startups // 4) + ["Series A"] * (num_startups // 2) + ["Series B"] * (num_startups // 4)
    stages = stages[:num_startups] + ["Series A"] * (num_startups - len(stages))
    rng.shuffle(stages)

    data = []

    for i in range(len(stages)):
        if stages[i] == "Seed":
            yoy_growth = rng.uniform(1.20, 3.00)
            nrr = rng.uniform(0.95, 1.15)
            customer_growth = yoy_growth * rng.uniform(0.8, 1.1)
            tam_bn = rng.uniform(0.5, 3.0)

            runway = rng.uniform(4, 14)
            churn = rng.uniform(0.08, 0.25)
            rev_concentration = rng.uniform(0.20, 0.60)

            burn_rate_k = rng.uniform(40, 100)
            ltv_cac = rng.uniform(1.5, 3.5)
            payback_months = rng.uniform(12, 24)
            rev_per_emp = rng.uniform(50000, 100000)
            gross_margin = rng.uniform(0.55, 0.75)

        elif stages[i] == "Series A":
            yoy_growth = rng.uniform(0.60, 1.50)
            nrr = rng.uniform(1.05, 1.30)
            customer_growth = yoy_growth * rng.uniform(0.8, 1.0)
            tam_bn = rng.uniform(2.0, 10.0)

            runway = rng.uniform(10, 20)
            churn = rng.uniform(0.05, 0.12)
            rev_concentration = rng.uniform(0.10, 0.35)

            burn_rate_k = rng.uniform(120, 300)
            ltv_cac = rng.uniform(3.0, 5.5)
            payback_months = rng.uniform(8, 15)
            rev_per_emp = rng.uniform(100000, 180000)
            gross_margin = rng.uniform(0.70, 0.85)

        else:  # Series B
            yoy_growth = rng.uniform(0.30, 0.75)
            nrr = rng.uniform(1.15, 1.40)
            customer_growth = yoy_growth * rng.uniform(0.7, 0.9)
            tam_bn = rng.uniform(5.0, 25.0)

            runway = rng.uniform(14, 28)
            churn = rng.uniform(0.02, 0.07)
            rev_concentration = rng.uniform(0.03, 0.15)

            burn_rate_k = rng.uniform(250, 600)
            ltv_cac = rng.uniform(4.0, 7.0)
            payback_months = rng.uniform(5, 12)
            rev_per_emp = rng.uniform(150000, 250000)
            gross_margin = rng.uniform(0.75, 0.88)
        
        pmf_indicator = rng.integers(4, 11) if stages[i] != "Seed" else rng.integers(2, 8)
        market_regulatory_risk = rng.integers(1, 10)
        execution_risk = rng.integers(2, 9)

        strat_alignment = rng.integers(3, 11)
        strat_synergies = rng.integers(2, 11)
        strat_positioning = rng.integers(4, 11)
        future_fundraising = rng.integers(3, 11)
        competitive_diff = rng.integers(3, 11)

        data.append(
            {
                'id': i,
                'stage': stages[i],
                'yoy_revenue_growth': round(yoy_growth, 4),
                'nrr': round(nrr, 4),
                'customer_growth_rate': round(customer_growth, 4),
                'tam_bn': round(tam_bn, 2),
                "pmf_indicator": pmf_indicator,
                'runway_months': round(runway, 1),
                'customer_churn': round(churn, 4),
                'revenue_concentration_risk': round(rev_concentration, 4),
                "market_regulatory_risk": market_regulatory_risk,
                "execution_risk": execution_risk,
                'monthly_burn_rate_k': round(burn_rate_k, 1),
                'ltv_cac': round(ltv_cac, 2),
                'cac_payback_months': round(payback_months, 1),
                'revenue_per_employee': int(rev_per_emp),
                'gross_margin': round(gross_margin, 4),
                "strategic_alignment": strat_alignment,
                "portfolio_synergies": strat_synergies,
                "market_positioning": strat_positioning,
                "future_fundraising_potential": future_fundraising,
                "competitive_differentiation": competitive_diff
            }
        )

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_portfolio(seed=42, num_startups=20)
    df.to_csv('portfolio/mock_portfolio.csv', index=False)

