"""
FIGURES OF THE RUNTIME TESTS

Draws the figures of the wiki page "Performance experiments" from the results of
runtime_tests_vi.py, runtime_tests_pi.py and runtime_tests_mpi.py (results/*.csv):
boxplots of the relative runtime of pymdptoolbox (its runtime divided by MDPSolver's)
for each sparsity and number of states, with MDPSolver in parallel (its default) and
on one thread.

usage: python3 plot_results.py [FOLDER]   (default: ../Images)
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"vi": "Value Iteration", "pi": "Policy Iteration", "mpi": "Modified Policy Iteration"}
COLORS = ("#66c2a5", "#fc8d62", "#8da0cb")
PANELS = (("RelativeDifference", "MDPSolver in parallel (default)"),
          ("RelativeDifferenceOneThread", "MDPSolver on one thread"))


def figure(name, df, folder):
    states = sorted(df["NumStates"].unique())
    sparsities = sorted(df["Sparsity"].unique())
    width = 0.8 / len(states)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (column, title) in zip(axes, PANELS):
        for i, sp in enumerate(sparsities):
            for j, ns in enumerate(states):
                values = df[(df["Sparsity"] == sp) & (df["NumStates"] == ns)][column]
                ax.boxplot(values, positions=[i - 0.4 + width * (j + 0.5)], widths=width * 0.9,
                           patch_artist=True, boxprops=dict(facecolor=COLORS[j], edgecolor="#444444"),
                           medianprops=dict(color="#444444"), whiskerprops=dict(color="#444444"),
                           capprops=dict(color="#444444"), flierprops=dict(markeredgecolor="#444444"))
            if i > 0:
                ax.axvline(i - 0.5, color="#dddddd", linewidth=0.8)
        ax.axhline(1, color="black", linestyle="--", linewidth=1)
        ax.set_xticks(range(len(sparsities)))
        ax.set_xticklabels(["%g" % (100 * sp) for sp in sparsities])
        ax.set_xlim(-0.5, len(sparsities) - 0.5)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Sparsity (%)")
        ax.set_ylabel("Relative Runtime of pymdptoolbox")
        ax.set_title(title)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#444444") for c in COLORS[:len(states)]]
    fig.legend(handles, [str(ns) for ns in states], title="Number of states",
               loc="upper right", bbox_to_anchor=(0.995, 0.93))
    fig.legend([plt.Line2D([], [], color="black", linestyle="--", linewidth=1)], ["MDPSolver Runtime"],
               loc="upper right", bbox_to_anchor=(0.995, 0.72))
    fig.suptitle("%s - Relative Runtime (pymdptoolbox vs. MDPSolver)" % NAMES[name])
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    fig.savefig(os.path.join(folder, "%s_test.png" % name), dpi=100)
    plt.close(fig)


def main():
    folder = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else os.path.join(HERE, "..", "Images")
    for name in NAMES:
        path = os.path.join(HERE, "results", "mdp_test_results_%s.csv" % name)
        if os.path.exists(path):
            figure(name, pd.read_csv(path), folder)
            print("wrote", os.path.join(folder, "%s_test.png" % name))


if __name__ == "__main__":
    main()
