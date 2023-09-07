""" Simple script to plot metrics vs safety threshold. """

import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":

    df = pd.read_csv("plots/data/cql_safety_threshold.csv")

    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(df["safety_threshold"], df["safe_episode_length"])
    ax[0].set_xlabel("Safety threshold")
    ax[0].set_ylabel("Safe episode length")
    ax[0].set_title("Safe episode length vs safety threshold")
    ax[1].plot(df["safety_threshold"], df["explore_fraction"])
    ax[1].set_xlabel("Safety threshold")
    ax[1].set_ylabel("Explore fraction")
    ax[1].set_title("Explore fraction vs safety threshold")
    fig.set_tight_layout(True)
    fig.show()
    input("Press Enter to continue...")

    fig.savefig("plots/plots/safety_threshold.png")