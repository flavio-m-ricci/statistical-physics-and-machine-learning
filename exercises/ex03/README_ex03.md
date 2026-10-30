# Exercise 03 — Randomized Perceptron Learning

**Course:** Statistical Physics and Machine Learning, Sapienza University of Rome  
**Instructor:** Prof. Chiara Cammarota  
**Student implementation and analysis:** Flavio Massimiliano Ricci

> **About this exercise.** This README independently restates the experimental specification from a course assignment. It is not a copy of the original instructor-provided document; the original PDF is not included.

## Objective

Study the generalization of a *student* perceptron trained on labels supplied by a fixed *teacher* perceptron. The learning rule introduces a random multiplicative factor in each weight update. The experiment examines how test error depends on training-set size $P$ and input dimension $N$.

## Teacher–student setup

Each input $\boldsymbol S\in\{-1,+1\}^{N}$ encodes two binary integers of $N/2$ bits each (in its first and second halves). The target label specifies which integer is larger:

$$
y=\begin{cases}
+1,&n_T>n_B,\\
-1,&n_T\le n_B.
\end{cases}
$$

The teacher has fixed weights $\boldsymbol T=\boldsymbol J^*$ defined by

$$
J_i^*=2^{N/2-i}\quad\left(1\le i\le\frac N2\right),\qquad
J_i^*=-J_{i-N/2}^*\quad\left(\frac N2+1\le i\le N\right).
$$

For an input $\boldsymbol S^\mu$, its target label is the teacher's classification,

$$
y_T^\mu=\operatorname{sign}\!\left(\sum_{i=1}^{N}T_iS_i^\mu\right),
$$

with equal-number cases labeled $-1$, as required by the exercise. The student starts with weights drawn from $\mathcal N(0,1)$.

## Randomized update rule

When the student's prediction differs from the teacher's label, update each component as

$$
J_i(t+1)=J_i(t)+\frac{1}{\sqrt N}\,y_T^\mu S_i^\mu(1+r_i),
$$

where the random factors satisfy

$$
r_i\sim\mathcal N(0,50).
$$

Here **50 is the variance**, not the standard deviation. Apply the rule over successive passes through the training examples, checking the training error after each pass, until the training error is zero.

## Experiment design

Run the full experiment at the following sizes:

| Input dimension | Number of bits per integer | Training-set sizes $P$ |
|---:|---:|---|
| $N=20$ | 10 | 1, 10, 20, 50, 100, 150, 200 |
| $N=40$ | 20 | 1, 2, 20, 40, 100, 200, 300 |

For **each** combination of $N$ and $P$:

1. Generate $P$ training inputs and label them using the teacher.
2. Initialize the student's weights with Gaussian random values.
3. Train using the randomized misclassification update until the training error vanishes.
4. Compute the fraction of misclassified examples on a **fresh test set of $P_{\mathrm{test}}=1000$** cases.
5. Repeat the experiment **1,000 times** and estimate the mean test error and its error bar.

## Plots and interpretation requested

- Plot mean test error $\epsilon$ versus training-set size $P$ for $N=20$.
- Repeat the study for $N=40$ and compare both curves on the same plot.
- Explore a re-expression or rescaling of the horizontal variable that causes the two datasets to approximately collapse onto a common curve; discuss the choice and its limitations.

The assignment does not prescribe a particular error-bar estimator or random seed. Those implementation details should be stated in the student analysis for reproducibility.

## Files and execution

The Python implementation, student's written solution and generated results are stored within this exercise directory. Exact run commands and figure links should be added after checking the filenames and runtime requirements of the actual script; they are intentionally not guessed here.

## Attribution

Exercise topic and experimental settings: *Statistical Physics and Machine Learning*, Sapienza University of Rome, Prof. Chiara Cammarota. Description rewritten for an academic code portfolio; original course handout not redistributed.
