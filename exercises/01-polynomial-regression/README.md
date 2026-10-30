# Exercise 01 — Polynomial Regression

**Course:** Statistical Physics and Machine Learning, Sapienza University of Rome  
**Instructor:** Prof. Chiara Cammarota  
**Student implementation and analysis:** Flavio Massimiliano Ricci

> **About this exercise.** This is an independently worded description of a course assignment. It preserves the mathematical problem and experimental requirements but does not reproduce the instructor's original handout. The Python code, figures and student report, where included, are separate student work.

## Objective

Investigate how the degree of a polynomial model, the amount of training data and the observation noise influence fitting and predictions on previously unseen inputs.

## Mathematical setup

Consider scalar input–output observations $(x_\mu,y_\mu)$ generated according to

$$
y_\mu=f(x_\mu)+\eta_\mu,
$$

where $x_\mu$ is sampled uniformly from the specified input interval. The noise is Gaussian, with zero mean and covariance

$$
\mathbb{E}[\eta_\mu]=0,\qquad
\mathbb{E}[\eta_\mu\eta_\nu]=\sigma^2\delta_{\mu\nu}.
$$

Use each of the following data-generating functions:

$$
f_A(x)=2x,\qquad
f_B(x)=2x-10x^5+15x^{10}.
$$

Fit polynomials of degree $d\in\{1,3,10\}$:

$$
p_d(x;\boldsymbol\theta)=\sum_{k=0}^{d}\theta_kx^k.
$$

Thus the three models have 2, 4 and 11 adjustable coefficients, respectively.

## Required experiments

Perform the complete fitting and testing procedure separately for **both** $f_A$ and $f_B$ in each setting:

| Experiment | Training points $P$ | Noise $\sigma$ | Training input range | Test points $P_{\mathrm{test}}$ | Test input range |
|---|---:|---:|---|---:|---|
| A | 10 | 0 | $[0,1]$ | 20 | $[0,1.25]$ |
| B | 100 | 1 | $[0,1]$ | 20 | $[0,1.25]$ |
| C | 10,000 | 1 | $[0,1]$ | 100 | $[0,1.25]$ |

For every experiment:

1. Generate the training sample from the selected data-generating function.
2. Fit polynomial models of degrees **1**, **3** and **10** to the training observations.
3. Independently generate a test sample using the same noise level as the training sample, but with test inputs spanning $[0,1.25]$.
4. Compare model predictions with the underlying function and with the new observations.
5. Discuss the fit quality and the behavior beyond the training interval (in particular, $x>1$).

## Questions to investigate

- How does the match between model complexity and the true generating function affect the predictions?
- What changes when noise is introduced?
- How do the results change as the number of training observations increases?
- How does performance on training observations compare with performance on the test set?

## Files and execution

- [`polynomial_regression.py`](polynomial_regression.py): Python implementation of the exercise.
- `training_*.png` and `test_*.png` (when present): visualizations generated for the different experiments.
- [`report.pdf`](report.pdf): student report/solution, if retained in this directory.

From this directory, with the repository's Python dependencies installed:

```bash
python polynomial_regression.py
```

**Reproducibility note:** The repository should document any random seeds, fitting routine and numerical-library versions used to regenerate its saved figures. The README describes the assignment; it does not assert that all experimental conclusions have been independently validated.

## Attribution

Exercise topic and numerical specifications: *Statistical Physics and Machine Learning*, Sapienza University of Rome, Prof. Chiara Cammarota. This description is rewritten for a public student portfolio; the original instructor-provided PDF is not redistributed here.
