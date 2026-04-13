import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")
sns.axes_style("darkgrid")


def plot_dists(
    df,
    fig_title,
    x_col="water_balance",
    hue_col="dataset",
    x_label="Water Balance (m)",
    out_file = None,
    show_fig = False
):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    sns.histplot(
        df,
        x=x_col,
        hue=hue_col,
        log_scale=True,
        element="step",
        fill=False,
        cumulative=True,
        stat="density",
        common_norm=False,
        ax=ax1,
    )
    sns.kdeplot(df, x=x_col, hue=hue_col, log_scale=True, ax=ax2, legend=False)
    fig.suptitle(fig_title)
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)

    if show_fig:
        plt.show()

    if out_file is not None:
        plt.savefig(out_file)
