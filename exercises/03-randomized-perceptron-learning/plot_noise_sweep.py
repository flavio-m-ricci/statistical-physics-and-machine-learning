from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_variances(variances, n=40):
    folder = Path(__file__).parent / "results"

    for var in variances:
        name = "0.0" if var == 0 else str(var)
        data = np.loadtxt(folder / f"N{n}_results_var{name}.txt")
        P, mean, sem = data[:, 0], data[:, 1], data[:, 4]

        plt.errorbar(
            P / n,
            mean,
            yerr=sem,
            fmt="o-",
            capsize=3,
            label=r"$\sigma^2$" + f" = {var}",
        )

    plt.xlabel(r"$\alpha = P/N$")
    plt.ylabel(r"$\langle \epsilon \rangle$")
    plt.title(r"Generalization error vs $\alpha$" + f"— N = {n}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(folder / f"variance_comparison_N{n}.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_variances([0, 0.01, 1, 5, 20, 50, 200, 500])
