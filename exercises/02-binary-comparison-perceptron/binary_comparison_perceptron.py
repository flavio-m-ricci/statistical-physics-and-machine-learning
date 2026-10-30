import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# seed
rng = np.random.default_rng(seed=12)


def generate_single_input():  # generate a random input of 20 elements {1, -1}
    x = rng.integers(0, 2, size=20)
    return 2 * x - 1


def generate_input(n):  # generate a random input of n vectors of 20 elements {1, -1}
    return 2 * rng.integers(0, 2, size=(n, 20)) - 1


def perceptron(
    J, s
):  # general perceptron: takes a 20 elements vector, the synaptic weights and gives the result
    return 1 if np.dot(J, s) > 0 else -1


# define the perfect cuplings
J_perf = np.zeros(20)
for i in range(10):
    J_perf[i] = 2 ** (9 - i)
    J_perf[i + 10] = -J_perf[i]


def perfect_perceptron(
    s,
):  # perfect perceptron: takes a 20 elements vector and gives the result
    return perceptron(J_perf, s)


def decode(bits):  # decode a binary vector of 10 elements into an integer
    n = 0
    for bit in bits:
        n = 2 * n + int((bit + 1) // 2)
    return n


def benchmark(
    s,
):  # benchmark: takes a 20 elements vector and gives the correct result // indidependent of the perfect perceptron
    n_T = decode(s[:10])
    n_B = decode(s[10:])

    return 1 if n_T > n_B else -1


def evaluate_benchmark_error_rate(
    inputs,
):  # evaluate the error rate of the perfect perceptron with respect to the benchmark
    error = 0.0
    for s in inputs:
        vero = benchmark(s)
        sper = perfect_perceptron(s)
        if vero != sper:
            error += 1.0
    return error / len(inputs)


def generate_random_couplings():  # generate a random synaptic couplings vector of 20 elements
    J = rng.normal(loc=0, scale=1, size=20)
    return J


def evaluate_error_rate(
    inputs, labels, J
):  # evaluate the error rate of a general perceptron with respect to the perfect perceptron (optimized)
    predictions = np.where(inputs @ J > 0, 1, -1)
    return np.mean(predictions != labels)


def evaluate_true_label_dataset(
    inputs,
):  # evaluate true labels of a dataset with perfect perceptron
    return np.where(inputs @ J_perf > 0, 1, -1)


def train_perceptron_single(
    inputs, labels, J
):  # train a perceptron with the given inputs and labels and return the training error rate
    eta = 1 / np.sqrt(20)  # learning rate

    for s, l in zip(inputs, labels):
        if perceptron(J, s) != l:
            J += eta * s * l  # update the synaptic weights

    return evaluate_error_rate(inputs, labels, J)


def plot_couplings(J, error_rate, number_of_examples):
    fig = plt.figure(figsize=(8, 5))
    plt.xlabel("i")
    plt.ylabel(r"$\mathrm{sign}(J_i^*J_i)\mathrm{log}_{2}(|J_i|)$")
    plt.title(
        rf"Couplings of the perceptron with $P={number_of_examples}$, "
        rf"$\epsilon={error_rate:.4f}$"
    )
    plt.xticks(range(1, 21))
    plt.yticks(range(-5, 10, 1))
    plt.grid()

    y = np.sign(J_perf * J) * np.log2(np.abs(J))
    plt.scatter(range(1, 11), y[:10], label=r"$1 \leq i \leq 10$", color="blue")
    plt.scatter(range(11, 21), y[10:], label=r"$11 \leq i \leq 20$", color="red")

    plt.legend()
    plt.savefig(
        FIGURES_DIR / f"couplings_{number_of_examples}_err_{error_rate}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()
    plt.close(fig)

# SIMULATION


# 1 and 2: generate 10000 random inputs and evaluate the error rate of the perfect perceptron

inputs = generate_input(10000)
error_rate_pfpr = evaluate_benchmark_error_rate(inputs)
print("Error rate of the perfect perceptron: ", error_rate_pfpr)


# 3 and 4: assign random couplings and evaluate the error rate of the random perpectron (with respect to the perfect perceptron)

inputs = generate_input(10000)
J = generate_random_couplings()
true_labels = evaluate_true_label_dataset(inputs)
error_rate_ranpr = evaluate_error_rate(inputs, true_labels, J)
print("Error rate of the random perceptron: ", error_rate_ranpr)


# 5: produce two sets of examples P=500 , P=2000 and their correct classification.

inputs_500 = generate_input(500)
inputs_2000 = generate_input(2000)
labels_500 = evaluate_true_label_dataset(inputs_500)
labels_2000 = evaluate_true_label_dataset(inputs_2000)


# 6 and 7: training of the percetron with the two sets across the entire dataset and evaluation of the training error rate. Stop when the error rate is exactrly zero on the training set. Study the evolution of the couplings and plot them with respect to the perfect couplings.

J_500 = generate_random_couplings()
error_rate_500 = evaluate_error_rate(inputs_500, labels_500, J_500)
plot_couplings(J_500, error_rate_500, 500)

condition_1 = False
condition_2 = False
while True:
    error_rate_500 = train_perceptron_single(inputs_500, labels_500, J_500)

    if error_rate_500 < 0.1 and not condition_1:
        plot_couplings(J_500, error_rate_500, 500)
        condition_1 = True

    if error_rate_500 < 0.03 and not condition_2:
        plot_couplings(J_500, error_rate_500, 500)
        condition_2 = True

    if error_rate_500 == 0:
        plot_couplings(J_500, error_rate_500, 500)
        break

print("Training error rate of the perceptron with 500 examples: ", error_rate_500)


J_2000 = generate_random_couplings()
error_rate_2000 = evaluate_error_rate(inputs_2000, labels_2000, J_2000)
plot_couplings(J_2000, error_rate_2000, 2000)

condition_1 = False
condition_2 = False
while True:
    error_rate_2000 = train_perceptron_single(inputs_2000, labels_2000, J_2000)

    if error_rate_2000 < 0.1 and not condition_1:
        plot_couplings(J_2000, error_rate_2000, 2000)
        condition_1 = True

    if error_rate_2000 < 0.03 and not condition_2:
        plot_couplings(J_2000, error_rate_2000, 2000)
        condition_2 = True

    if error_rate_2000 == 0:
        plot_couplings(J_2000, error_rate_2000, 2000)
        break

print("Training error rate of the perceptron with 2000 examples: ", error_rate_2000)


# 8: evaluate the error rate of the perceptron trained with 500 examples and 2000 examples on a new set of random inputs

inputs = generate_input(10000)
true_labels = evaluate_true_label_dataset(inputs)

error_rate_500_test = evaluate_error_rate(inputs, true_labels, J_500)
print("Error rate of the perceptron trained with 500 examples: ", error_rate_500_test)

error_rate_2000_test = evaluate_error_rate(inputs, true_labels, J_2000)
print("Error rate of the perceptron trained with 2000 examples: ", error_rate_2000_test)
