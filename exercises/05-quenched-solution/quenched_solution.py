from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from scipy.special import log_ndtr
from scipy.integrate import quad
from functools import lru_cache

# -----------------------------------------------
# EXECUTION PARAMETERS
# -----------------------------------------------

PARAMETRIC_CURVE = False  # If True, compute the parametric curve and plot it.
PLOT_R_CONVERGENCE = False  # plot R_i vs i for f_1 and f_2
SOLVE_EQUATION = False  # solve the equation for all combinations of alpha, R, a and f
SHOW_RESULTS_EQUATION = False  # show the results of the equation (in a csv file)
SHOW_HEATMAPS = False  # show heat maps of the convergence results
SET_AUTOMATIC_PROCEDURE = False  # set the automatic procedure for the iterative solver
PLOT_FINAL_EPSILON = True  # plot epsilon vs alpha for the automatic procedure, the parametric curve and the numerical solution of code 03

# -----------------------------------------------
# SETUP
# -----------------------------------------------

FOLDER = Path(__file__).parent
RESULTS_03_FOLDER = FOLDER / "results_03"
RESULTS = FOLDER / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES = FOLDER / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------
# FUNCTIONS
# -----------------------------------------------


def log_h(u):
    """
    The function returns log(H(u))
    We evaluate H(u) in the following way: considering phi, the cumulative distribution function of the standard normal distribution, H(u) = 1 - phi(u) = phi(-u)
    we choose to evaluate the logaritm because when u is large, H(u) is very small and we can have numerical issues. So instead of evaluating 1/H in the I integral, we evaluate exp(-log(H(u))) which is more stable.
    """
    return log_ndtr(-u)


@lru_cache(
    maxsize=4096
)  # Cache the results of the i_integral function, since it is called multiple times for R=0.5 for example
def i_integral(R):
    """
    The function returns the I(R) integral as a function of R.
    R must be in the range [0,1)
    """
    if not 0 <= R < 1:
        raise ValueError("R must be in the range [0,1)")

    sqrt_R = np.sqrt(R)

    def integrand(x):
        return np.exp(-(1 + R) * x**2 / 2 - np.log(2 * np.pi) / 2 - log_h(sqrt_R * x))

    result, error = quad(integrand, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10)
    return result


def f1(R, alpha):  # returns f1
    return alpha / np.pi * np.sqrt(1 - R) * i_integral(R)


def f2(R, alpha):  # returns f2
    return 1 - (np.pi * R / (alpha * i_integral(R))) ** 2


# -----------------------------------------------
# PARAMETRIC CURVE
# -----------------------------------------------


def param_curve_coordinates(R_values):
    """
    Returns the parametric curve coordinates, with R identified as the parameter t (R=t).
    Remember R ranges from 0 to 1.
    Takes a list of R values as input
    """

    alpha_values = np.array(
        [np.pi * R / (np.sqrt(1 - R) * i_integral(R)) for R in R_values]
    )

    epsilon_values = np.arccos(R_values) / np.pi

    return alpha_values, epsilon_values


def plot_param_curve(alpha_values, epsilon_values):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(alpha_values, epsilon_values, label="Parametric Curve", color="blue")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\epsilon$")
    ax.legend()
    ax.grid()
    plt.title("Parametric Curve of " + r"$\epsilon$" + " vs " + r"$\alpha$")

    fig.savefig(FIGURES / "parametric_curve.png", dpi=300)
    plt.show()
    plt.close(fig)


# -----------------------------------------------
# ITERATIVE SOLVER
# -----------------------------------------------


class IterativeSolver:
    def __init__(
        self, alpha_schedule, R_0_schedule, a_schedule, func, tol=1e-10, n_max=1000
    ):
        self.alpha_schedule = alpha_schedule
        self.R_0_schedule = R_0_schedule
        self.a_schedule = a_schedule
        self.func = func
        self.tol = tol
        self.n_max = n_max

    def iterate(
        self, alpha, R_0, a
    ):  # for a given alpha, initial R_0 and interpolation parameter a, iterate until convergence or max iterations
        R = R_0
        history = [R]
        residual = np.nan  # initialize residual to NaN

        # Check initial condition
        if not np.isfinite(R) or not 0 <= R < 1:
            return R, history, False, 0, residual

        for _ in range(self.n_max):
            R_new = (1 - a) * R + a * self.func(R, alpha)
            history.append(R_new)

            # check numerical and physical validity of R
            if not np.isfinite(R_new) or not 0 <= R_new < 1:
                return (
                    R_new,
                    history,
                    False,
                    len(history) - 1,
                    residual,
                )  # false if not converged

            residual = abs(R_new - self.func(R_new, alpha))

            if not np.isfinite(residual):
                return (
                    R_new,
                    history,
                    False,
                    len(history) - 1,
                    residual,
                )  # false if not converged

            if residual < self.tol:
                return (
                    R_new,
                    history,
                    True,
                    len(history) - 1,
                    residual,
                )  # true if converged

            R = R_new

        return (
            R,
            history,
            False,
            len(history) - 1,
            residual,
        )  # false if not converged

    def test_iterative_procedure(
        self,
    ):  # test the iterative procedure for all combinations of alpha, R_0 and a
        results = {
            "alpha": [],
            "R_0": [],
            "a": [],
            "R": [],
            "converged": [],
            "iterations": [],
            "residual": [],
        }

        for alpha in tqdm(self.alpha_schedule, desc="alpha"):
            for R_0 in tqdm(self.R_0_schedule, desc="R_0", leave=False):
                for a in tqdm(self.a_schedule, desc="a", leave=False):
                    R, _, converged, iterations, residual = self.iterate(alpha, R_0, a)
                    results["alpha"].append(alpha)
                    results["R_0"].append(R_0)
                    results["a"].append(a)
                    results["R"].append(R)
                    results["converged"].append(converged)
                    results["iterations"].append(iterations)
                    results["residual"].append(residual)

        df = pd.DataFrame(results)
        df.to_csv(
            RESULTS / f"test_iterative_procedure_results_{self.func.__name__}.csv",
            index=False,
        )

    def plot_R_convergence(self, alpha, R_0, a):  # plot R_i vs i
        _, history, converged, _, _ = self.iterate(alpha, R_0, a)

        fig = plt.figure(figsize=(8, 5))

        plt.plot(history, label=r"$R_i$", color="blue")

        message = " (converged)" if converged else " (not converged)"
        plt.title(
            rf"$R_i$ vs $i$ — "
            rf"$\alpha={alpha}$, $R_0={R_0}$, $a={a}$" + message
        )
        plt.xlabel(r"$i$")
        plt.ylabel(r"$R_i$")
        plt.legend()
        plt.savefig(
            FIGURES
            / f"R_convergence_{self.func.__name__}_alpha_{alpha}_R0_{R_0}_a_{a}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

    def check_convergence_results(
        self,
    ):  # check the convergence results and print the summary
        dfs = []

        for func in [f1, f2]:
            df = pd.read_csv(
                RESULTS / f"test_iterative_procedure_results_{func.__name__}.csv"
            )

            df["function"] = func.__name__
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # successful runs
        df["success"] = (
            df["converged"]
            & df["R"].between(0, 1)
            & np.isfinite(df["R"])
            & np.isfinite(df["residual"])
            & (df["residual"] < self.tol)
        )

        # Summary of results for different functions, R_0 and a
        summary = df.groupby(["function", "R_0", "a"]).agg(
            success_rate=("success", "mean"),
            mean_residual=("residual", "mean"),
        )

        summary[
            "mean_iterations"
        ] = (  # mediate only the successful ones (that were not interrupted by numerical issues)
            df[df["success"]].groupby(["function", "R_0", "a"])["iterations"].mean()
        )

        # write summary to csv
        summary.to_csv(RESULTS / "convergence_summary_raw.csv")

        # Rank configurations by success rate, mean iterations and mean residual (in this order)
        summary = summary.sort_values(
            by=["success_rate", "mean_residual", "mean_iterations"],
            ascending=[False, True, True],
        )
        summary.to_csv(RESULTS / "convergence_summary_ranked.csv")

    def plot_heatmaps(self):  # plot heat maps of the convergence results
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)
        summary = pd.read_csv(
            RESULTS / "convergence_summary_raw.csv", index_col=[0, 1, 2]
        )

        for ax, func in zip(axes, ["f1", "f2"]):
            heatmap = summary.loc[func]["success_rate"].unstack("R_0")

            im = ax.imshow(
                heatmap,
                origin="lower",
                aspect="auto",
                cmap="viridis",
                vmin=0,
                vmax=1,
            )

            ax.set_xticks(range(len(heatmap.columns)))
            ax.set_xticklabels(
                [f"{x:.2f}" for x in heatmap.columns],
                rotation=45,
            )

            ax.set_yticks(range(len(heatmap.index)))
            ax.set_yticklabels([f"{y:.2f}" for y in heatmap.index])

            ax.set_title(func)
            ax.set_xlabel(r"$R_0$")
            ax.set_ylabel(r"$a$")

        fig.colorbar(im, ax=axes, label="Success rate")

        fig.savefig(
            FIGURES / "convergence_success_heatmaps.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

    def plot_success_rates(
        self,
    ):  # plots the success rates of the iterative procedure for f1 and f2 as a function of alpha
        rates = {}

        for func in ["f1", "f2"]:
            df = pd.read_csv(RESULTS / f"test_iterative_procedure_results_{func}.csv")

            rates[func] = df.groupby("alpha")["converged"].mean()

        rates = pd.DataFrame(rates)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 5))

        for func in ["f1", "f2"]:
            ax.plot(rates.index, rates[func], label=func)

        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel("Success rate")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid()

        fig.savefig(FIGURES / "success_rate_vs_alpha.png", dpi=300)
        plt.show()
        plt.close(fig)

        return rates

    def select_best_methods(
        self, rates
    ):  # selects where to use f1 and f2 based on the success rates of the iterative procedure and which configuration of R_0 and a to use for each function

        # Region where both functions always converge
        overlap = rates[(rates["f1"] == 1) & (rates["f2"] == 1)]

        if overlap.empty:
            raise RuntimeError("No overlap found")

        alpha_switch = (overlap.index.min() + overlap.index.max()) / 2

        best_configs = {}

        for func in ["f1", "f2"]:
            df = pd.read_csv(RESULTS / f"test_iterative_procedure_results_{func}.csv")

            # Select the corresponding alpha range
            if func == "f1":
                df = df[df["alpha"] <= alpha_switch]
            else:
                df = df[df["alpha"] > alpha_switch]

            # Evaluate each configuration
            summary = df.groupby(["R_0", "a"]).agg(
                success_rate=("converged", "mean"),
                mean_residual=("residual", "mean"),
                mean_iterations=("iterations", "mean"),
            )

            # Keep configurations with 100% success
            summary = summary[summary["success_rate"] == 1]

            if summary.empty:
                raise RuntimeError(f"No suitable configuration for {func}")

            # Choose the configuration with the smallest residual,
            # then the smallest number of iterations
            summary = summary.sort_values(
                by=["mean_residual", "mean_iterations"],
                ascending=[True, True],
            )

            best_configs[func] = summary.index[0]

        print(f"Switching alpha: {alpha_switch:.6f}")
        print(f"Best configurations: {best_configs}")

        return alpha_switch, best_configs

    def automatic_procedure(
        self, alpha_switch, best_configs
    ):  # automatic procedure to determine epsilon vs alpha using the best configuration for f1 and f2

        results = {
            "alpha": [],
            "R": [],
            "epsilon": [],
            "converged": [],
            "iterations": [],
            "residual": [],
        }

        for alpha in tqdm(self.alpha_schedule, desc="alpha"):
            if alpha == 0:
                R, converged, iterations, residual = 0.0, True, 0, 0.0

            else:
                if alpha <= alpha_switch:
                    self.func = f1
                    R_0, a = best_configs["f1"]
                else:
                    self.func = f2
                    R_0, a = best_configs["f2"]

                R, _, converged, iterations, residual = self.iterate(alpha, R_0, a)

                if not converged:
                    raise RuntimeError(f"Convergence failed for alpha={alpha}")

            results["alpha"].append(alpha)
            results["R"].append(R)
            results["epsilon"].append(np.arccos(R) / np.pi)
            results["converged"].append(converged)
            results["iterations"].append(iterations)
            results["residual"].append(residual)

        df = pd.DataFrame(results)

        df.to_csv(
            RESULTS / "automatic_procedure_results.csv",
            index=False,
        )


# -----------------------------------------------
# FINAL PLOT OF EPSILON VS ALPHA
# -----------------------------------------------


def plot_final_epsilon_vs_alpha():  # plot epsilon vs alpha for the automatic procedure, the parametric curve and the numerical solution of code 03
    df_automatic = pd.read_csv(RESULTS / "automatic_procedure_results.csv")
    data_03 = np.loadtxt(RESULTS_03_FOLDER / "N20_results_var50.txt")
    x_03 = data_03[:, 0] / 20
    y_03 = data_03[:, 1]

    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        alpha_values,
        epsilon_values,
        label="Parametric curve",
        color="red",
        linestyle="-",
        # linewidth=3,
        zorder=1,
    )
    plt.plot(
        df_automatic["alpha"],
        df_automatic["epsilon"],
        label="Iterative procedure",
        color="blue",
        linestyle="--",
        # linewidth=1.8,
        zorder=2,
    )
    plt.plot(
        x_03,
        y_03,
        label="Numerical solution (code 03)",
        color="green",
        linestyle=":",
        marker="o",
        zorder=3,
    )

    plt.xlim(0, 10)
    plt.ylim(0, 0.5)

    plt.title(r"$\epsilon$ vs $\alpha$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\epsilon$")
    plt.legend()
    plt.savefig(FIGURES / "final_epsilon_vs_alpha.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# -----------------------------------------------
# MAIN
# -----------------------------------------------

if __name__ == "__main__":
    if PARAMETRIC_CURVE or PLOT_FINAL_EPSILON:
        R_values = np.linspace(0, 0.999, 100)  # R ranges from 0 to 1
        alpha_values, epsilon_values = param_curve_coordinates(R_values)

    if PARAMETRIC_CURVE:
        plot_param_curve(alpha_values, epsilon_values)

    if (
        PLOT_R_CONVERGENCE
        or SOLVE_EQUATION
        or SHOW_RESULTS_EQUATION
        or SHOW_HEATMAPS
        or SET_AUTOMATIC_PROCEDURE
    ):
        solver = IterativeSolver(
            np.linspace(0, 10, 100),
            np.linspace(0.01, 0.99, 10),
            np.linspace(0.1, 0.9, 10),
            f1,
        )

    # plot R_i vs i for f1 and f2 for alpha= [5.0. 0.1], R_0=0.5 and a=0.5
    if PLOT_R_CONVERGENCE:
        for alpha in [5.0, 0.1]:  # f1
            solver.plot_R_convergence(alpha=alpha, R_0=0.5, a=0.5)

        solver.func = f2
        for alpha in [5.0, 0.1]:  # f2
            solver.plot_R_convergence(alpha=alpha, R_0=0.5, a=0.5)

    # test the iterative procedure for all combinations of alpha, epsilon_0 and a
    if SOLVE_EQUATION:
        solver.func = f1
        solver.test_iterative_procedure()  # f1

        solver.func = f2
        solver.test_iterative_procedure()  # f2

    # Collect the results of the equation for f1 and f2
    if SHOW_RESULTS_EQUATION:
        solver.check_convergence_results()

    # Show heat maps of the convergence results
    if SHOW_HEATMAPS:
        solver.plot_heatmaps()

    # Automatic procedure to determine epsilon vs alpha using the best configuration
    if SET_AUTOMATIC_PROCEDURE:
        rates = solver.plot_success_rates()

        alpha_switch, best_configs = solver.select_best_methods(rates)

        solver.automatic_procedure(alpha_switch, best_configs)

        print("alpha_switch: ", alpha_switch, "\nBest configurations: ", best_configs)

    # Final plot of epsilon vs alpha for the automatic procedure, the parametric curve and the numerical solution of code 03
    if PLOT_FINAL_EPSILON:
        plot_final_epsilon_vs_alpha()
