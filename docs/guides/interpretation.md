# Interpreting Results

## Sharp Null Test Results

The sharp null hypothesis $H_0: Y(1,m) = Y(0,m)$ for all $m$ states that the treatment effect on the outcome operates entirely through the mediator $M$.

- **`reject = True`**: Evidence that some treatment effect bypasses $M$ (direct effect exists)
- **`p_value`**: Probability of observing the test statistic under $H_0$
- **Method**: CS (recommended for empirical work), ARP, FSST

## Lower Bound Results

`lb_frac_affected` returns the minimum fraction of always-takers whose outcome is affected by treatment through channels other than $M$.

- **`lower_bound = 0.11`** means at least 11% of units with $M(1)=M(0)=m$ have direct effects
- Higher values provide stronger evidence against full mediation

## ADE Bounds

`bounds_ade_ats` provides Lee-style trimming bounds on the Average Direct Effect for always-takers.

- **`[lower_bound, upper_bound]`** brackets the true ADE
- If both bounds have the same sign, the direction of the direct effect is identified

## Partial Density

Shows the density decomposition that drives the lower bound calculation.

- Gap between treated and control density for a mediator value indicates evidence of direct effects
- Shaded area represents the identified minimum fraction affected
