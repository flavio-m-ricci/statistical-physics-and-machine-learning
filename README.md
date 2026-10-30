
# Statistical Physics and Machine Learning

**Computational Exercises | Sapienza University of Rome**

**Author:** Flavio Massimiliano Ricci  
**Programme:** MSc in Physics — Statistical Physics and Complexity  
**Course:** Statistical Physics and Machine Learning  
**Instructor:** Prof. Chiara Cammarota

## Overview

This repository contains my Python implementations, numerical experiments, and written reports for the *Statistical Physics and Machine Learning* course at Sapienza University of Rome.

The projects explore fundamental concepts at the intersection of statistical physics and machine learning, including polynomial regression, model complexity, generalization, perceptron learning, stochastic optimization, and the statistical properties of learning algorithms.

In addition to the prescribed coursework, selected exercises include supplementary computational investigations exploring the influence of model parameters on learning behavior.

## Exercises

### 01 — Polynomial Regression

**Topics:** Supervised learning, polynomial fitting, model complexity, overfitting, generalization, and extrapolation.

This exercise investigates polynomial regression on synthetic datasets generated from two different functions.

Polynomial models of different degrees are fitted under varying conditions of training-set size and observation noise. Their predictions are compared with the underlying functions, with particular attention to the differences between interpolation and extrapolation beyond the training domain.

**[View Exercise 01](exercises/01-polynomial-regression/README.md)**

### 02 — Binary Comparison with a Perceptron

**Topics:** Binary classification, linear classifiers, perceptron learning, weight optimization, and generalization error.

This exercise studies a perceptron designed to compare two binary integers.

An analytically constructed reference classifier is compared with randomly initialized and trained perceptrons. Numerical experiments investigate the evolution of synaptic weights, the reduction of training error, and the relationship between training-set size and generalization performance.

**[View Exercise 02](exercises/02-binary-comparison-perceptron/README.md)**

### 03 — Randomized Perceptron Learning

**Topics:** Teacher–student learning, stochastic weight updates, statistical learning, generalization error, and finite-size scaling.

This exercise investigates a teacher–student perceptron model in which learning updates are affected by multiplicative Gaussian noise.

The prescribed investigation examines generalization error as a function of training-set size and input dimension, using repeated independent experiments to estimate mean errors and statistical uncertainties.

#### Supplementary Investigation — Noise-Variance Dependence

Beyond the original assignment, I extended the numerical study to investigate how the variance of the stochastic learning updates influences generalization performance.

The original exercise specifies a noise variance of 50. The additional experiments consider several values, including the noise-free limit and variances above and below the prescribed value.

These results allow a comparative investigation of stochastic learning dynamics and their dependence on noise intensity. The supplementary study is explicitly distinguished from the original coursework.

**[View Exercise 03](exercises/03-randomized-perceptron-learning/README.md)**

## Repository Structure

```text
statistical-physics-and-machine-learning/
├── README.md
├── requirements.txt
├── .gitignore
└── exercises/
    ├── 01-polynomial-regression/
    │   ├── README.md
    │   ├── polynomial_regression.py
    │   ├── report.pdf
    │   ├── simulation_results.txt
    │   └── figures/
    │
    ├── 02-binary-comparison-perceptron/
    │   ├── README.md
    │   ├── binary_comparison_perceptron.py
    │   ├── report.pdf
    │   └── figures/
    │
    └── 03-randomized-perceptron-learning/
        ├── README.md
        ├── randomized_perceptron.py
        ├── plot_noise_sweep.py
        ├── report.pdf
        └── results/
```

Each exercise includes its own README describing the mathematical formulation, objectives, experimental methodology, and numerical analysis.

## Technologies and Computational Methods

The projects use Python and scientific computing libraries, primarily:

- **NumPy:** Numerical linear algebra, random sampling, and array operations.
- **Matplotlib:** Scientific visualization and statistical plots.
- **Numba:** JIT compilation for computationally intensive simulations.
- **tqdm:** Progress monitoring for repeated numerical experiments.

The implementations involve supervised learning, iterative optimization, pseudorandom simulations, statistical error estimation, and numerical analysis.

## Installation and Reproducibility

### Requirements

Install the required Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

### Running the Exercises

Each exercise can be executed independently using its corresponding Python script.

**Exercise 01 — Polynomial Regression**

```bash
cd exercises/01-polynomial-regression
python polynomial_regression.py
```

**Exercise 02 — Binary Comparison Perceptron**

```bash
cd exercises/02-binary-comparison-perceptron
python binary_comparison_perceptron.py
```

**Exercise 03 — Randomized Perceptron Learning**

```bash
cd exercises/03-randomized-perceptron-learning
python randomized_perceptron.py
```

The third exercise uses configurable simulation parameters, including input dimension, number of repetitions, and noise variance. Some configurations are computationally intensive.

Specific simulation settings, execution instructions, and descriptions of saved outputs are provided in the individual exercise READMEs.

## Academic Attribution

The exercises originate from the *Statistical Physics and Machine Learning* course taught by Prof. Chiara Cammarota at Sapienza University of Rome.

The Python implementations, numerical investigations, generated figures, and accompanying reports are presented as my coursework.

The supplementary numerical experiments are identified separately from the original assignments.

The exercise descriptions are independently written and paraphrased for documentation purposes. The original instructor-provided assignment sheets are not redistributed.

This repository documents educational and computational work and is not presented as a collection of original research publications.
