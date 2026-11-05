from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# -----------------------------------------------
# EXECUTION PARAMETERS
# -----------------------------------------------

PARAMETRIC_CURVE = False  # plot the parametric curve of epsilon(alpha)
PLOT_EPSILON_CONVERGENCE = False  # plot epsilon_i vs i for f_1 and f_2
SOLVE_EQUATION = (
    False  # solve the equation for all combinations of alpha, epsilon, a and f
)
SHOW_RESULTS_EQUATION = False  # show the results of the equation (in a csv file)
SHOW_HEATMAPS = False  # show heat maps of the convergence results
SET_AUTOMATIC_PROCEDURE = False  # set the automatic procedure for the iterative solver
PLOT_FINAL_EPSILON = True  # plot epsilon vs alpha for the automatic procedure, the parametric curve and the numerical solution of code 03

# -----------------------------------------------
# PATH SETUP
# -----------------------------------------------

FOLDER = Path(__file__).parent
RESULTS_03_FOLDER = FOLDER / "results_03"
FIGURES_FOLDER = FOLDER / "figures"
FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS = FOLDER / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------
# PARAMETRIC CURVE
# -----------------------------------------------


def cot(x):  # cotangent function
    return 1 / np.tan(x)


def function(x):  # alpha(epsilon)
    return (1 - x) * np.pi * cot(np.pi * x)


def plot_param_curve():  # parametric plot of epsilon(alpha)
    fig = plt.figure(figsize=(8, 5))

    plt.plot(
        function(np.linspace(0.001, 0.5, 500)),
        np.linspace(0.001, 0.5, 500),
        label=r"$\epsilon(\alpha)$",
        color="red",
    )

    plt.title(r"Parametric curve of $\epsilon(\alpha)$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\epsilon$")
    plt.legend()
    plt.savefig(FIGURES_FOLDER / "parametric_curve.png", dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


# -----------------------------------------------
# SOLVE EQUATION
# -----------------------------------------------


class IterativeSolver:
    def __init__(
        self,
        alpha_schedule,
        epsilon_0_schedule,
        a_schedule,
        func,
        tol=1e-10,
        n_max=1000,
    ):
        self.alpha_schedule = alpha_schedule
        self.epsilon_0_schedule = epsilon_0_schedule
        self.a_schedule = a_schedule
        self.func = func  # as argument give: IterativeSolver.f_1 or IterativeSolver.f_2
        self.tol = tol
        self.n_max = n_max

    # METHOD 1

    @staticmethod
    def f_1(epsilon, alpha):
        return 1 - alpha * np.tan(np.pi * epsilon) / np.pi

    # METHOD 2

    @staticmethod
    def f_2(epsilon, alpha):
        return np.arctan(np.pi * (1 - epsilon) / alpha) / np.pi

    # ITERATIVE PROCEDURE

    def iterate(
        self, alpha, epsilon_0, a
    ):  # for a given alpha, initial epsilon_0 and interpolation parameter a, iterate until convergence or max iterations
        epsilon = epsilon_0
        history = [epsilon]
        residual = np.nan  # initialize residual to NaN

        # Check initial condition
        if (
            not np.isfinite(epsilon)
            or not 0 <= epsilon <= 0.5
            or (self.func == self.f_1 and epsilon == 0.5)
        ):
            return epsilon, history, False, 0, residual

        for _ in range(self.n_max):
            epsilon_new = (1 - a) * epsilon + a * self.func(epsilon, alpha)
            history.append(epsilon_new)

            # check numerical and physical validity of epsilon
            if (
                not np.isfinite(epsilon_new)
                or not 0 <= epsilon_new <= 0.5
                or (self.func == self.f_1 and epsilon_new == 0.5)
            ):
                return (
                    epsilon_new,
                    history,
                    False,
                    len(history) - 1,
                    residual,
                )  # false if not converged

            residual = abs(epsilon_new - self.func(epsilon_new, alpha))

            if not np.isfinite(residual):
                return (
                    epsilon_new,
                    history,
                    False,
                    len(history) - 1,
                    residual,
                )  # false if not converged

            if residual < self.tol:
                return (
                    epsilon_new,
                    history,
                    True,
                    len(history) - 1,
                    residual,
                )  # true if converged

            epsilon = epsilon_new

        return (
            epsilon,
            history,
            False,
            len(history) - 1,
            residual,
        )  # false if not converged

    def test_iterative_procedure(
        self,
    ):  # test the iterative procedure for all combinations of alpha, epsilon_0 and a
        results = {
            "alpha": [],
            "epsilon_0": [],
            "a": [],
            "epsilon": [],
            "converged": [],
            "iterations": [],
            "residual": [],
        }

        for alpha in tqdm(self.alpha_schedule, desc="alpha"):
            for epsilon_0 in tqdm(
                self.epsilon_0_schedule, desc="epsilon_0", leave=False
            ):
                for a in tqdm(self.a_schedule, desc="a", leave=False):
                    epsilon, _, converged, iterations, residual = self.iterate(
                        alpha, epsilon_0, a
                    )
                    results["alpha"].append(alpha)
                    results["epsilon_0"].append(epsilon_0)
                    results["a"].append(a)
                    results["epsilon"].append(epsilon)
                    results["converged"].append(converged)
                    results["iterations"].append(iterations)
                    results["residual"].append(residual)

        df = pd.DataFrame(results)
        df.to_csv(
            RESULTS / f"test_iterative_procedure_results_{self.func.__name__}.csv",
            index=False,
        )

    def plot_epsilon_convergence(self, alpha, epsilon_0, a):  # plot epsilon_i vs i
        _, history, converged, _, _ = self.iterate(alpha, epsilon_0, a)

        fig = plt.figure(figsize=(8, 5))

        plt.plot(history, label=r"$\epsilon_i$", color="blue")

        message = " (converged)" if converged else " (not converged)"
        plt.title(r"$\epsilon_i$" + " vs i" + message)
        plt.xlabel(r"$i$")
        plt.ylabel(r"$\epsilon_i$")
        plt.legend()
        plt.savefig(
            FIGURES_FOLDER / f"epsilon_convergence_{self.func.__name__}_a_{a}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()
        plt.close(fig)

    def check_convergence_results(
        self,
    ):  # check the convergence results and print the summary
        dfs = []

        for func in [self.f_1, self.f_2]:
            df = pd.read_csv(
                RESULTS / f"test_iterative_procedure_results_{func.__name__}.csv"
            )

            df["function"] = func.__name__
            dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)

        # successful runs
        df["success"] = (
            df["converged"]
            & df["epsilon"].between(0, 0.5)
            & np.isfinite(df["epsilon"])
            & np.isfinite(df["residual"])
            & (df["residual"] < self.tol)
        )

        # Summary of results for different functions, epsilon_0 and a
        summary = df.groupby(["function", "epsilon_0", "a"]).agg(
            success_rate=("success", "mean"),
            mean_residual=("residual", "mean"),
        )

        summary[
            "mean_iterations"
        ] = (  # mediate only the successful ones (that were not interrupted by numerical issues)
            df[df["success"]]
            .groupby(["function", "epsilon_0", "a"])["iterations"]
            .mean()
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

        for ax, func in zip(axes, ["f_1", "f_2"]):
            heatmap = summary.loc[func]["success_rate"].unstack("epsilon_0")

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
            ax.set_xlabel(r"$\epsilon_0$")
            ax.set_ylabel(r"$a$")

        fig.colorbar(im, ax=axes, label="Success rate")

        fig.savefig(
            FIGURES_FOLDER / "convergence_success_heatmaps.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.show()
        plt.close(fig)

    def automatic_procedure(
        self,
    ):  # automatic procedure to determine epsilon vs alpha using the best configuration
        summary = pd.read_csv(
            RESULTS / "convergence_summary_ranked.csv", index_col=[0, 1, 2]
        )

        best_config = summary.iloc[0]

        best_function, best_epsilon_0, best_a = best_config.name

        self.func = {
            "f_1": self.f_1,
            "f_2": self.f_2,
        }[best_function]

        results = {
            "alpha": [],
            "epsilon": [],
            "converged": [],
            "iterations": [],
            "residual": [],
        }

        for alpha in tqdm(self.alpha_schedule, desc="alpha"):
            epsilon, _, converged, iterations, residual = self.iterate(
                alpha, best_epsilon_0, best_a
            )
            results["alpha"].append(alpha)
            results["epsilon"].append(epsilon)
            results["converged"].append(converged)
            results["iterations"].append(iterations)
            results["residual"].append(residual)

        df = pd.DataFrame(results)
        df.to_csv(RESULTS / "automatic_procedure_results.csv", index=False)


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
        function(np.linspace(0.001, 0.5, 500)),
        np.linspace(0.001, 0.5, 500),
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
    )

    plt.xlim(0, 10)
    plt.ylim(0, 0.5)

    plt.title(r"$\epsilon$ vs $\alpha$")
    plt.xlabel(r"$\alpha$")
    plt.ylabel(r"$\epsilon$")
    plt.legend()
    plt.savefig(
        FIGURES_FOLDER / "final_epsilon_vs_alpha.png", dpi=300, bbox_inches="tight"
    )
    plt.show()
    plt.close(fig)


# -----------------------------------------------
# MAIN
# -----------------------------------------------

if __name__ == "__main__":
    # plot the parametric curve of epsilon(alpha)
    if PARAMETRIC_CURVE:
        plot_param_curve()

    if (
        PLOT_EPSILON_CONVERGENCE
        or SOLVE_EQUATION
        or SHOW_RESULTS_EQUATION
        or SHOW_HEATMAPS
        or SET_AUTOMATIC_PROCEDURE
    ):
        solver = IterativeSolver(
            np.linspace(0.1, 10, 100),
            np.linspace(0.01, 0.5, 10),
            np.linspace(0.1, 0.9, 10),
            IterativeSolver.f_1,
        )

    # plot epsilon_i vs i for f_1 and f_2 for alpha=5, epsilon_0=0.25 and a in [0.5, 0.9, 0.1]
    if PLOT_EPSILON_CONVERGENCE:
        for a in [0.5, 0.9, 0.1]:  # f_1
            solver.plot_epsilon_convergence(alpha=5, epsilon_0=0.25, a=a)

        solver.func = IterativeSolver.f_2
        for a in [0.5, 0.9, 0.1]:  # f_2
            solver.plot_epsilon_convergence(alpha=5, epsilon_0=0.25, a=a)

    # test the iterative procedure for all combinations of alpha, epsilon_0 and a
    if SOLVE_EQUATION:
        solver.func = IterativeSolver.f_1
        solver.test_iterative_procedure()  # f_1

        solver.func = IterativeSolver.f_2
        solver.test_iterative_procedure()  # f_2

    # Collect the results of the equation for f_1 and f_2
    if SHOW_RESULTS_EQUATION:
        solver.check_convergence_results()

    # Show heat maps of the convergence results
    if SHOW_HEATMAPS:
        solver.plot_heatmaps()

    # Automatic procedure to determine epsilon vs alpha using the best configuration
    if SET_AUTOMATIC_PROCEDURE:
        solver.automatic_procedure()

    # Final plot of epsilon vs alpha for the automatic procedure, the parametric curve and the numerical solution of code 03
    if PLOT_FINAL_EPSILON:
        plot_final_epsilon_vs_alpha()
