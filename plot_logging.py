import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set()


def parse_log_file(file_path):
    metrics = {
        "steps": [],
        "ranking_losses": [],
        "d_flops": [],
        "flops_losses": [],
        "doc_lengths": [],
        "d_rep_avgs": [],
        "d_rep_90s": [],
        # 'q_rep_avgs': [], 'q_rep_90s': []
    }

    pattern_dict = {
        "steps": r"Step (\d+)",
        "ranking_losses": r"ranking loss moving avg:([\d.]+)",
        "d_flops": r"d_flops: ([\d.]+)",
        "flops_losses": r"flops_loss: ([\d.]+)",
        "doc_lengths": r"avg doc length: ([\d.]+)",
        "d_rep_avgs": r"d_rep_avg: ([\d.]+)",
        "d_rep_90s": r"d_rep_90: ([\d.]+)",
        # 'q_rep_avgs': r'q_rep_avg: ([\d.]+)',
        # 'q_rep_90s': r'q_rep_90: ([\d.]+)'
    }

    with open(file_path, "r") as f:
        for line in f:
            if "Step" in line:
                for key, pattern in pattern_dict.items():
                    value = float(re.search(pattern, line).group(1))
                    metrics[key].append(value)

    return metrics


def create_subplot(ax, x, y, xlabel, ylabel, title, use_log=False):
    ax.plot(x, y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if use_log:
        ax.set_yscale("log")
    ax.grid(True)


if __name__ == "__main__":
    # Read the log file
    metrics = parse_log_file("path/to/log_file.txt")

    # Define plot configurations
    plot_configs = [
        ("ranking_losses", "Ranking Loss Moving Avg", "Ranking Loss vs Steps", False),
        ("d_flops", "D_FLOPS", "D_FLOPS vs Steps", True),
        ("flops_losses", "FLOPS Loss", "FLOPS Loss vs Steps", True),
        (
            "doc_lengths",
            "Average Document Length",
            "Average Document Length vs Steps",
            True,
        ),
        ("d_rep_avgs", "D_REP_AVG", "Document Representation Average vs Steps", False),
        (
            "d_rep_90s",
            "D_REP_90",
            "Document Representation 90th Percentile vs Steps",
            False,
        ),
        # ('q_rep_avgs', 'Q_REP_AVG', 'Query Representation Average vs Steps', False),
        # ('q_rep_90s', 'Q_REP_90', 'Query Representation 90th Percentile vs Steps', False)
    ]

    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()

    # Create all subplots
    for idx, (metric_key, ylabel, title, use_log) in enumerate(plot_configs):
        create_subplot(
            axes[idx],
            metrics["steps"],
            metrics[metric_key],
            "Steps",
            ylabel + " $\mathbf{log}$" if use_log else ylabel,
            title,
            use_log,
        )

    # Remove the extra subplot
    axes[-1].remove()

    # Adjust layout
    plt.tight_layout()
    plt.show()
