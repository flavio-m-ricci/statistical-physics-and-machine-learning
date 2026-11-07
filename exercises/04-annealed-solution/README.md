# Exercise 04 — Annealed Solution to Randomized Perceptron Learning

**Course:** Statistical Physics and Machine Learning, Sapienza University of Rome  
**Instructor:** Prof. Chiara Cammarota  
**Student implementation and analysis:** Flavio Massimiliano Ricci

## Overview

This exercise investigates the annealed prediction for the generalization error of a teacher–student perceptron. The implicit analytical relation is plotted parametrically and solved numerically using two fixed-point formulations.

The implementation compares their convergence, searches for reliable iteration parameters and overlays the resulting analytical curve with the numerical data from Exercise 03.

## Mathematical setup

A student perceptron with $N$ binary inputs receives $P=\alpha N$ labeled examples from a teacher perceptron. In the annealed approximation, the typical student with zero training error has generalization error $\epsilon(\alpha)$ satisfying

```math
\alpha=(1-\epsilon)\pi\cot(\pi\epsilon),
\qquad 0<\epsilon\leq\frac12.
```

Here $\alpha=P/N$ is the number of training examples per input dimension, and $\epsilon$ is the probability of disagreement with the teacher on a new example.

### Parametric representation

The equation directly specifies $\alpha$ as a function of $\epsilon$. The script therefore plots the curve parametrically, using $t=\epsilon$:

```math
\bigl(\alpha(t),\epsilon(t)\bigr)
=\left((1-t)\pi\cot(\pi t),\,t\right).
```

The numerical plot uses 500 values of $t$ between 0.001 and 0.5, avoiding the singular endpoint $t=0$. The limit at $\alpha=0$ is $\epsilon=0.5$.

## Fixed-point methods

To obtain $\epsilon$ at a prescribed value of $\alpha$, the implicit equation is rewritten in two equivalent forms:

```math
f_1(\epsilon,\alpha)
=1-\frac{\alpha}{\pi}\tan(\pi\epsilon),
```

```math
f_2(\epsilon,\alpha)
=\frac{1}{\pi}\arctan\!\left(\frac{\pi(1-\epsilon)}{\alpha}\right).
```

For either function, the relaxed fixed-point iteration is

```math
\epsilon_{i+1}
=(1-a)\epsilon_i+a\,f_j(\epsilon_i,\alpha),
\qquad j\in\{1,2\}.
```

The relaxation parameter $a$ controls how strongly each iteration follows the fixed-point map. Iteration stops successfully when

```math
\left|\epsilon_i-f_j(\epsilon_i,\alpha)\right|<10^{-10}.
```

The solver also checks that successive values are finite and remain in the physical range $0\leq\epsilon\leq0.5$. It reports failure if these checks are violated or if 1,000 iterations are reached. For $f_1$, the singular starting value $\epsilon_0=0.5$ is explicitly rejected.

## Experimental design

### Prescribed convergence tests

Both maps are tested with $\alpha=5$, $\epsilon_0=0.25$ and three values of the relaxation parameter:

| Map | $\alpha$ | $\epsilon_0$ | Relaxation $a$ |
|---|---:|---:|---|
| $f_1$ | 5 | 0.25 | 0.1, 0.5, 0.9 |
| $f_2$ | 5 | 0.25 | 0.1, 0.5, 0.9 |

For each case, the script saves a plot of $\epsilon_i$ against the iteration number $i$.

### Parameter sweep

A broader numerical investigation varies the target ratio, initial guess and relaxation parameter:

| Parameter | Values |
|---|---|
| $\alpha$ | 100 equally spaced values in $[0.1,10]$ |
| $\epsilon_0$ | 10 equally spaced values in $[0.01,0.5]$ |
| $a$ | 10 equally spaced values in $[0.1,0.9]$ |
| Convergence tolerance | $10^{-10}$ |
| Maximum iterations | 1,000 |

This gives 10,000 trials per fixed-point map. For each trial, the script records the final estimate, convergence status, number of iterations and fixed-point residual.

The results are summarized by map, initial guess and relaxation parameter. The configurations are ranked by decreasing success rate, then increasing mean residual, then increasing mean iteration count among successful trials. A heatmap visualizes the success rates.

### Automatic solution

The script selects the highest-ranked configuration from the saved summary and uses it to solve for $\epsilon(\alpha)$ at the same 100 values of $\alpha\in[0.1,10]$.

The final figure compares three datasets:

1. The analytical parametric curve.
2. The automatically computed fixed-point solutions.
3. The saved Exercise 03 numerical results for $N=20$ and update-noise variance $v=50$, plotted against $\alpha=P/N$.

Although `results_03/` also contains the $N=40$ data, the current final plotting function reads only `N20_results_var50.txt`. The numerical comparison does not plot uncertainty bars.

## Main findings

The saved results illustrate three main points:

- **Convergence depends on the fixed-point map.** The $f_1$ iteration can leave the physical interval or fail to converge, particularly at larger relaxation parameters. The $f_2$ formulation is more stable over the tested configurations.
- **Robust automatic solution.** All 100 combinations of $\epsilon_0$ and $a$ tested with $f_2$ converge for every sampled $\alpha$. The highest-ranked saved configuration uses $f_2$, $\epsilon_0=0.5$ and $a=0.9$. Its automatic sweep converges at all 100 values of $\alpha$, with residuals below $10^{-10}$.
- **Comparison with learning simulations.** The parametric and iterative calculations describe the same annealed relation. The Exercise 03 measurements follow the same qualitative decrease of generalization error with increasing $\alpha$, but need not coincide exactly: they are finite-dimensional results obtained with a specific teacher and a noisy training procedure.

The difference in convergence behavior can be understood from the slopes of the two maps:

```math
\frac{\partial f_1}{\partial\epsilon}
=-\alpha\sec^2(\pi\epsilon),
\qquad
\frac{\partial f_2}{\partial\epsilon}
=-\frac{\alpha}{\alpha^2+\pi^2(1-\epsilon)^2}.
```

The first map can have a large slope in magnitude, while the second remains contractive over the physical interval for $\alpha>0$. The relaxation parameter further affects the local stability of the iteration.

The [report](report.pdf) accompanies the numerical analysis. Full sweep results and convergence summaries are available in [`results/`](results/).

## Figures and data

| Contents | File |
|---|---|
| Analytical parametric curve | [Figure](figures/parametric_curve.png) |
| $f_1$ convergence, $a=0.1$ | [Figure](figures/epsilon_convergence_f_1_a_0.1.png) |
| $f_1$ convergence, $a=0.5$ | [Figure](figures/epsilon_convergence_f_1_a_0.5.png) |
| $f_1$ convergence, $a=0.9$ | [Figure](figures/epsilon_convergence_f_1_a_0.9.png) |
| $f_2$ convergence, $a=0.1$ | [Figure](figures/epsilon_convergence_f_2_a_0.1.png) |
| $f_2$ convergence, $a=0.5$ | [Figure](figures/epsilon_convergence_f_2_a_0.5.png) |
| $f_2$ convergence, $a=0.9$ | [Figure](figures/epsilon_convergence_f_2_a_0.9.png) |
| Parameter-sweep success rates | [Heatmaps](figures/convergence_success_heatmaps.png) |
| Annealed prediction and Exercise 03 data | [Final figure](figures/final_epsilon_vs_alpha.png) |
| Individual $f_1$ trials | [CSV](results/test_iterative_procedure_results_f_1.csv) |
| Individual $f_2$ trials | [CSV](results/test_iterative_procedure_results_f_2.csv) |
| Convergence summaries | [Raw](results/convergence_summary_raw.csv) · [Ranked](results/convergence_summary_ranked.csv) |
| Automatic fixed-point solutions | [CSV](results/automatic_procedure_results.csv) |

## Files and execution

| Path | Contents |
|---|---|
| [`annealed_solution.py`](annealed_solution.py) | Parametric curve, fixed-point solvers, convergence tests, parameter sweep and final comparison |
| [`figures/`](figures/) | Saved plots of the analytical curve, iterations, heatmaps and final comparison |
| [`results/`](results/) | Numerical sweep, summaries and automatic solutions |
| [`results_03/`](results_03/) | Copies of the numerical results and figures from Exercise 03 used for comparison |
| [`report.pdf`](report.pdf) | Student report |

From the repository root, install the dependencies and run:

```bash
python -m pip install -r requirements.txt pandas
python exercises/04-annealed-solution/annealed_solution.py
```

Alternatively, from this exercise directory:

```bash
python annealed_solution.py
```

### Execution switches

The main script uses Boolean switches near the beginning of `annealed_solution.py`:

| Switch | Operation |
|---|---|
| `PARAMETRIC_CURVE` | Save and display the parametric curve |
| `PLOT_EPSILON_CONVERGENCE` | Generate the six prescribed convergence plots |
| `SOLVE_EQUATION` | Run the full parameter sweep for both maps |
| `SHOW_RESULTS_EQUATION` | Read the sweep results and write raw and ranked summary CSV files |
| `SHOW_HEATMAPS` | Plot success-rate heatmaps from the raw summary |
| `SET_AUTOMATIC_PROCEDURE` | Select the top-ranked configuration and save its automatic solutions |
| `PLOT_FINAL_EPSILON` | Compare the analytical curve, automatic solutions and Exercise 03 data |

**With the current defaults, only `PLOT_FINAL_EPSILON = True`.** Running the script therefore regenerates the final figure from the existing CSV files; it does not repeat the parameter sweep or the automatic solution.

To regenerate the full analysis, enable the relevant switches. The operations execute in the order shown in the script: convergence tests, parameter sweep, summary, heatmaps, automatic procedure and final comparison. The summary requires the two sweep CSV files; the heatmap requires the raw summary, and the automatic procedure requires the ranked summary. The final plot requires `automatic_procedure_results.csv` and `results_03/N20_results_var50.txt`.

Figures and results are saved relative to the exercise directory, and files with matching names are overwritten. Plotting functions call `plt.show()`, so a graphical Matplotlib backend may pause execution until figure windows are closed. For unattended plotting, set `MPLBACKEND=Agg`.

### Reproducibility

The annealed calculations and fixed-point iterations are deterministic; they do not use random sampling. Reproducing the stored outputs requires keeping the parameter grids, tolerance, iteration limit and ranking rule unchanged.

The final comparison uses precomputed stochastic results from Exercise 03. The current automated scan excludes $\alpha=0$, where the analytical endpoint is $\epsilon(0)=0.5$ and the expression for $f_2$ requires separate handling.

## Attribution

The exercise topic and numerical tasks originate from the *Statistical Physics and Machine Learning* course taught by Prof. Chiara Cammarota at Sapienza University of Rome.

The Python implementation, numerical investigations, figures and report are presented as student coursework. The exercise description is independently written, and the original instructor-provided assignment sheet is not redistributed.
