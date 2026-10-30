lazy import matplotlib.pyplot as plt
lazy import numpy as np
lazy import tqdm

# seed
rng = np.random.default_rng(seed=12)


def generate_input(p, n):  # generate a random input of p vectors of n elements {1, -1}
    return 2 * rng.integers(0, 2, size=(p, n)) - 1


def perceptron(
    J, s
):  # general perceptron: takes a n elements vector, the synaptic weights and gives the result
    return 1 if np.dot(J, s) > 0 else -1


# define the perfect cuplings
def generate_perfect_couplings(n):
    J_perf = np.zeros(n)
    for i in range(int(n // 2)):
        J_perf[i] = 2 ** ((n / 2) - 1 - i)
        J_perf[i + int(n // 2)] = -J_perf[i]
    return J_perf


def perfect_perceptron(
    J_perf, s
):  # perfect perceptron: takes a n elements vector and gives the result
    return perceptron(J_perf, s)


def generate_random_couplings(
    n,
):  # generate a random synaptic couplings vector of n elements
    J = rng.normal(loc=0, scale=1, size=n)
    return J


def evaluate_error_rate(
    inputs, labels, J
):  # evaluate the error rate of a general perceptron with respect to the perfect perceptron (optimized)
    predictions = np.where(inputs @ J > 0, 1, -1)
    return np.mean(predictions != labels)


def evaluate_true_label_dataset(
    inputs, J_perf
):  # evaluate true labels of a dataset with perfect perceptron
    return np.where(inputs @ J_perf > 0, 1, -1)


def train_perceptron_single(
    inputs, labels, J, n, variance=50
):  # train a perceptron with the given inputs and labels and return the training error rate
    eta = 1 / np.sqrt(n)  # learning rate

    for s, l in zip(inputs, labels):
        if perceptron(J, s) != l:
            J += (
                eta * s * l * (1 + rng.normal(loc=0.0, scale=np.sqrt(variance), size=n))
            )  # update the synaptic weights

    return evaluate_error_rate(inputs, labels, J)


def train_perceptron_multiple(
    inputs, labels, J, n, variance=50
):  # train a perceptron with the given inputs and labels for a number of epochs and return the training error rate
    while True:
        error = train_perceptron_single(inputs, labels, J, n, variance)
        if error == 0.0:
            break


def generate_examples(p, n, J_perf):
    inputs = generate_input(p, n)
    labels = evaluate_true_label_dataset(inputs, J_perf)
    return inputs, labels


# function for the simulation


def run_simulation(n, P, runs, dim_p):
    # perfect perceptron
    J_perf = generate_perfect_couplings(n)

    error = np.zeros(dim_p)
    mean_error_sq = np.zeros(dim_p)

    for k in tqdm.tqdm(range(runs)):
        # generate random J
        J = np.array([generate_random_couplings(n) for _ in range(dim_p)])

        # generate examples and train
        for p, i in zip(P, range(dim_p)):
            inputs, labels = generate_examples(p, n, J_perf)
            train_perceptron_multiple(inputs, labels, J[i], n)

        # test
        # generate test sets
        inputs_test, labels_test = zip(
            *(generate_examples(1000, n, J_perf) for _ in range(dim_p))
        )  # creates 7 arrays of (1000,n) and (1000,) respectively
        inputs_test = np.stack(inputs_test)  # creates a (7,1000,n) array
        labels_test = np.stack(labels_test)  # creates a (7,1000) array
        # evaluate error rate on test set
        error_rate_test = np.array(
            [
                evaluate_error_rate(inputs_test[i], labels_test[i], J[i])
                for i in range(dim_p)
            ]
        )

        error += error_rate_test / runs
        mean_error_sq += error_rate_test**2 / runs

    sample_variance = (mean_error_sq - error**2) * runs / (runs - 1)
    std = np.sqrt(np.maximum(sample_variance, 0))
    sem = np.std / np.sqrt(runs)  # standard error of the mean

    np.savetxt(
        f"N{n}_results.txt",
        np.column_stack((P, error, sample_variance, std, sem)),
        header="P mean_error variance std sem",
        fmt="%.10g",
    )


# SIMULATION

RUN_N20 = True
RUN_N40 = False
RUNS = 1000

P_N20 = [1, 10, 20, 50, 100, 150, 200]
P_N40 = [1, 2, 20, 40, 100, 200, 300]
DIM_P = 7

if RUN_N20:
    # Run N=20 simulation
    # Calculate mean, variance, SD, SEM
    # Save results to N20_results.txt
    n = 20

    run_simulation(n, P_N20, RUNS, DIM_P)

if RUN_N40:
    # Run N=40 simulation
    # Calculate mean, variance, SD, SEM
    # Save results to N40_results.txt
    n = 40

    run_simulation(n, P_N40, RUNS, DIM_P)
