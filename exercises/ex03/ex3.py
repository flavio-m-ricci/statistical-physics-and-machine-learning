from pathlib import Path
import numpy as np
from numba import njit
from tqdm.auto import tqdm


# ==========================================
# CONFIGURATION
# ==========================================

RUN_N20 = False
RUN_N40 = True

RUNS = 1000
SEED = 123
TEST_SIZE = 1000

NOISE_VARIANCE = 500
MAX_EPOCHS = 100_000

SAVE_EVERY = 25
PLOT_RESULTS = False

P_N20 = np.array([1, 10, 20, 50, 100, 150, 200])
P_N40 = np.array([1, 2, 20, 40, 100, 200, 300])

RESULTS_DIR = Path(__file__).resolve().parent / "ex3_results"


# ==========================================
# TEACHER AND DATA GENERATION
# ==========================================


def generate_perfect_couplings(n):
    half = 2.0 ** np.arange(n // 2 - 1, -1, -1)
    return np.concatenate((half, -half))


def generate_examples(p, n, J_perf, rng):
    inputs = (2 * rng.integers(0, 2, size=(p, n)) - 1).astype(np.int8)

    labels = np.where(inputs @ J_perf > 0, 1, -1).astype(np.int8)

    return inputs, labels


# ==========================================
# NUMBA-COMPILED TRAINING
# ==========================================


@njit(cache=True)
def train_perceptron(inputs, labels, J, rng, variance, max_epochs):
    p, n = inputs.shape

    eta = 1.0 / np.sqrt(n)
    noise_sd = np.sqrt(variance)

    for epoch in range(max_epochs):
        # One complete training cycle
        for mu in range(p):
            field = 0.0

            for i in range(n):
                field += J[i] * inputs[mu, i]

            prediction = 1 if field > 0.0 else -1

            if prediction != labels[mu]:
                for i in range(n):
                    J[i] += (
                        eta
                        * labels[mu]
                        * inputs[mu, i]
                        * (1.0 + rng.normal(0.0, noise_sd))
                    )

        # Evaluate training error after the cycle
        n_errors = 0

        for mu in range(p):
            field = 0.0

            for i in range(n):
                field += J[i] * inputs[mu, i]

            prediction = 1 if field > 0.0 else -1

            if prediction != labels[mu]:
                n_errors += 1

        if n_errors == 0:
            return epoch + 1

    return -1


# ==========================================
# SAVE RESULTS
# ==========================================


def save_results(n, P, mean, m2, completed):

    RESULTS_DIR.mkdir(exist_ok=True)

    if completed > 1:
        variance = m2 / (completed - 1)
    else:
        variance = np.full(len(P), np.nan)

    std = np.sqrt(np.maximum(variance, 0.0))
    sem = std / np.sqrt(completed)

    path = RESULTS_DIR / f"N{n}_results_var{NOISE_VARIANCE}.txt"

    np.savetxt(
        path,
        np.column_stack((P, mean, variance, std, sem)),
        header=(f"N={n}, completed_runs={completed}\nP mean_error variance std sem"),
        fmt=["%d", "%.10g", "%.10g", "%.10g", "%.10g"],
    )

    return path


# ==========================================
# MONTE CARLO SIMULATION
# ==========================================


def run_simulation(n, P, runs, rng):

    J_perf = generate_perfect_couplings(n)

    mean = np.zeros(len(P))
    m2 = np.zeros(len(P))

    for k in tqdm(range(1, runs + 1), desc=f"N={n}_var{NOISE_VARIANCE}", unit="run"):
        # Shared independent test dataset for this run
        inputs_test, labels_test = generate_examples(TEST_SIZE, n, J_perf, rng)

        weights = np.empty((len(P), n))

        # Train each student perceptron
        for j, p in enumerate(P):
            inputs, labels = generate_examples(int(p), n, J_perf, rng)

            weights[j] = rng.normal(loc=0.0, scale=1.0, size=n)

            epochs = train_perceptron(
                inputs,
                labels,
                weights[j],
                rng,
                NOISE_VARIANCE,
                MAX_EPOCHS,
            )

            if epochs == -1:
                if k > 1:
                    save_results(n, P, mean, m2, k - 1)

                raise RuntimeError(
                    f"N={n}, P={p}, run={k}: training failed after {MAX_EPOCHS} epochs"
                )

        # Evaluate all students using the same test set
        predictions = inputs_test @ weights.T > 0.0  # (1000,n) @ (7,n)^T -> (1000,7)
        correct_labels = (
            labels_test[:, None] > 0
        )  # (1000,) -> (1000,1) for broadcasting

        errors = np.mean(
            predictions != correct_labels, axis=0
        )  # average only over 1000, result (7,)

        # Welford's online mean and variance accumulation
        delta = errors - mean
        mean += delta / k
        m2 += delta * (errors - mean)

        # Periodic checkpoints
        if k % SAVE_EVERY == 0 or k == runs:
            save_results(n, P, mean, m2, k)

    print(f"N={n}: completed {runs} repetitions")


# ==========================================
# PLOTTING FROM SAVED RESULTS
# ==========================================


def plot_saved_results():

    import matplotlib.pyplot as plt

    # Load previously saved results
    results = {}

    for n in (20, 40):
        path = RESULTS_DIR / f"N{n}_results_var{NOISE_VARIANCE}.txt"

        if path.exists():
            data = np.loadtxt(path, ndmin=2)

            results[n] = {
                "P": data[:, 0],
                "mean": data[:, 1],
                "sem": data[:, 4],
            }
        else:
            print(f"No saved results found for N={n}")

    # ==========================================
    # PLOT 1: ERROR VS P FOR N=20
    # ==========================================

    if 20 in results:
        P = results[20]["P"]
        mean = results[20]["mean"]
        sem = results[20]["sem"]

        plt.figure(figsize=(8, 5))

        plt.errorbar(
            P,
            mean,
            yerr=sem,
            fmt="o-",
            capsize=4,
            label="N = 20",
        )

        plt.xlabel(r"$P$")
        plt.ylabel(r"$\langle \epsilon \rangle$")
        plt.title("Generalization error — N = 20")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(RESULTS_DIR / f"error_N20_var{NOISE_VARIANCE}.png", dpi=300)

    # ==========================================
    # PLOT 2: ERROR VS P FOR N=20 AND N=40
    # ==========================================

    if 20 in results and 40 in results:
        plt.figure(figsize=(8, 5))

        for n in (20, 40):
            P = results[n]["P"]
            mean = results[n]["mean"]
            sem = results[n]["sem"]

            plt.errorbar(
                P,
                mean,
                yerr=sem,
                fmt="o-",
                capsize=4,
                label=f"N = {n}",
            )

        plt.xlabel(r"$P$")
        plt.ylabel(r"$\langle \epsilon \rangle$")
        plt.title("Generalization error — N = 20 and N = 40")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(RESULTS_DIR / f"error_comparison_var{NOISE_VARIANCE}.png", dpi=300)

        # ==========================================
        # PLOT 3: DATA COLLAPSE
        # ==========================================

        plt.figure(figsize=(8, 5))

        for n in (20, 40):
            P = results[n]["P"]
            mean = results[n]["mean"]
            sem = results[n]["sem"]

            alpha = P / n

            plt.errorbar(
                alpha,
                mean,
                yerr=sem,
                fmt="o-",
                capsize=4,
                label=f"N = {n}",
            )

        plt.xlabel(r"$\alpha = P/N$")
        plt.ylabel(r"$\langle \epsilon \rangle$")
        plt.title("Generalization error — Superposition of N = 20 and N = 40")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        plt.savefig(RESULTS_DIR / f"data_collapse_var{NOISE_VARIANCE}.png", dpi=300)

    plt.show()


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    if RUN_N20:
        run_simulation(20, P_N20, RUNS, np.random.default_rng(SEED + 20))

    if RUN_N40:
        run_simulation(40, P_N40, RUNS, np.random.default_rng(SEED + 40))

    if PLOT_RESULTS:
        plot_saved_results()
