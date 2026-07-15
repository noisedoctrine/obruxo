# Experiment 13 Math Rendering Fixture

This temporary fixture isolates the Markdown parsing collision seen in the
Experiment 13 plan.

## Broken `$$` form with a standalone equals line

$$
E_{i,d,s}
=
\min_{a\in A_{d,s},\,\phi,\,g}
\left\|r_{i,d}-gS_\phi(a)\right\|_\infty.
$$

## Correct fenced `math` form

```math
E_{i,d,s}
=
\min_{a\in A_{d,s},\,\phi,\,g}
\left\|r_{i,d}-gS_\phi(a)\right\|_\infty.
```

## Representative corrected syntax

```math
\mathrm{resolved}^{13B}_{i,d,s}
=
\mathbf{1}\!\left[E^+_{i,d,s}(a)\leq\tau_{\mathrm{finish}}\right].
```

```math
(\phi_i(a),g_i(a))
=
\arg\min_{\phi,g}\ell\!\left(r_i,gS_{\phi_i}(a)\right),
\qquad
a^*=\arg\max_{a\in\mathcal C}\Delta_i(a).
```
