import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_data(dataframe: pd.DataFrame, env_name: str, ax: plt.Axes, label: str = None):

    max = dataframe[f'env_id: {env_name} - Metrics/EpRet__MAX']
    min = dataframe[f'env_id: {env_name} - Metrics/EpRet__MIN']
    mean = dataframe[f'env_id: {env_name} - Metrics/EpRet']
    steps = dataframe['TotalEnvSteps']

    ax.plot(steps, mean, label=label)
    ax.fill_between(steps, min, max, alpha=0.2)
    ax.set_title(f'{env_name}')

if __name__ == "__main__":
    cpo_df = pd.read_csv("plots/data/cpo_ep_ret.csv")
    ppolag_df = pd.read_csv("plots/data/ppolag_ep_ret.csv")

    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    fig.suptitle('Episode Return')

    envs = ['Walker2d-v4', 'Hopper-v4', 'Ant-v4']
    for env, ax in zip(envs, ax):
        plot_data(cpo_df, env, ax, label='CPO')
        plot_data(ppolag_df, env, ax, label='PPO-LAG')
        ax.legend()

    fig.set_tight_layout(True)
    fig.show()
    input("Press Enter to continue...")
