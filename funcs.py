import json
import numpy as np
import matplotlib.pyplot as plt



def save_to_json(name,dict):
    with open(f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(dict, f, ensure_ascii=False, indent=2)


def get_json(name):
    with open(f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def plot_cdf_wrt_label(df, column, x_max=None, title=None):
    """
    Plots the empirical CDF of a column
    split by label (0 vs 1).

    Parameters:
    -----------
    df : pandas.DataFrame
    column : str
        Name of numeric column to plot
    x_max : float or None
        Optional upper limit for x-axis
    title : str or None
        Optional custom title
    """

    # Split by label
    pos = df[df["label"] == 1][column].values
    neg = df[df["label"] == 0][column].values

    # Sort
    pos_sorted = np.sort(pos)
    neg_sorted = np.sort(neg)

    # Compute empirical CDF
    pos_cdf = (
        np.arange(1, len(pos_sorted) + 1) / len(pos_sorted)
        if len(pos_sorted) > 0 else None
    )

    neg_cdf = (
        np.arange(1, len(neg_sorted) + 1) / len(neg_sorted)
        if len(neg_sorted) > 0 else None
    )

    # Plot
    plt.figure()

    if pos_cdf is not None:
        plt.plot(pos_sorted, pos_cdf)

    if neg_cdf is not None:
        plt.plot(neg_sorted, neg_cdf)

    plt.xlabel(column)
    plt.ylabel("Cumulative Probability")

    if title:
        plt.title(title)
    else:
        plt.title(f"CDF of {column} by Label")

    plt.legend(["label = 1", "label = 0"])
    plt.grid(True)

    if x_max is not None:
        plt.xlim(0, x_max)

    plt.show()



def plot_pdf_by_label(df, column, bins=50, x_max=None, title=None):
    """
    Plots the empirical PDF (density histogram)
    of a column split by label (0 vs 1).
    """

    pos = df[df["label"] == 1][column].values
    neg = df[df["label"] == 0][column].values

    plt.figure()

    if len(pos) > 0:
        plt.hist(pos, bins=bins, density=True, histtype='step', label="label = 1")

    if len(neg) > 0:
        plt.hist(neg, bins=bins, density=True, histtype='step', label="label = 0")

    plt.xlabel(column)
    plt.ylabel("Density")

    if title:
        plt.title(title)
    else:
        plt.title(f"PDF of {column} by Label")

    if x_max is not None:
        plt.xlim(0, x_max)

    plt.legend()
    plt.grid(True)
    plt.show()









def plot_history_histogram(users_dict, bins=10):
    """
    Plots a histogram of the number of history messages per user.

    Parameters:
        users_dict (dict): Dictionary of users with a 'history' field
        bins (int): Number of histogram bins (default=10)
    """
    history_lengths = [
        len(user_data.get("history", []))
        for user_data in users_dict.values()
    ]

    if not history_lengths:
        print("No users or history data available.")
        return

    plt.figure()
    plt.hist(history_lengths, bins=bins)
    plt.xlabel("Number of History Messages")
    plt.ylabel("Number of Users")
    plt.title("Histogram of History Lengths per User")
    plt.show()





def plot_p_label_given_x(df, column, bins=50, x_max=None, title=None):
    """
    Plots P(label=1 | X) using binning.
    """

    x = df[column].values
    y = df["label"].values

    # Optional clipping
    if x_max is not None:
        mask = x <= x_max
        x = x[mask]
        y = y[mask]

    # Create bins
    bin_edges = np.linspace(np.min(x), np.max(x), bins + 1)
    bin_indices = np.digitize(x, bin_edges) - 1

    prob = []
    bin_centers = []

    for i in range(bins):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            p = np.mean(y[mask])  # fraction of label=1
            prob.append(p)
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

    plt.figure()
    plt.plot(bin_centers, prob)
    plt.xlabel(column)
    plt.ylabel("P(label=1 | X)")
    
    if title:
        plt.title(title)
    else:
        plt.title(f"P(label=1 | {column})")

    plt.grid(True)
    plt.show()
plot_p_label_given_x(df, "U-HA_R_RetweetPercent")