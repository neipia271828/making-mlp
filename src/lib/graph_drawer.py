import matplotlib.pyplot as plt
from pathlib import Path
from statistics import median


def _plot_series(ax, epochs, values, label):
    if not values:
        return

    ax.plot(epochs, values, marker="o", markersize=4, label=label)


def _summarize_series(values):
    if not values:
        return {"max": "", "min": "", "last": "", "last10_avg": "", "last10_median": ""}

    last_values = values[-10:]

    return {
        "max": f"{max(values):.4f}",
        "min": f"{min(values):.4f}",
        "last": f"{values[-1]:.4f}",
        "last10_avg": f"{sum(last_values) / len(last_values):.4f}",
        "last10_median": f"{median(last_values):.4f}",
    }


def _draw_summary_table(ax, title, train_values, valid_values):
    ax.axis("off")

    train_summary = _summarize_series(train_values)
    valid_summary = _summarize_series(valid_values)

    table = ax.table(
        cellText=[
            [
                "train",
                train_summary["max"],
                train_summary["min"],
                train_summary["last"],
                train_summary["last10_avg"],
                train_summary["last10_median"],
            ],
            [
                "valid",
                valid_summary["max"],
                valid_summary["min"],
                valid_summary["last"],
                valid_summary["last10_avg"],
                valid_summary["last10_median"],
            ],
        ],
        colLabels=["series", "max", "min", "last", "last10 avg", "last10 median"],
        cellLoc="center",
        loc="center",
        colWidths=[0.16, 0.16, 0.16, 0.16, 0.18, 0.18],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.08, 1.35)
    ax.set_title(title, fontsize=10, pad=6)


def graph_drawer(PROJECT, epochs, timestamp, train_losses, train_accuracies, valid_losses, valid_accuracies, show=True):
    graph_dir = Path(f"data/{PROJECT}/logs") / timestamp
    graph_dir.mkdir(parents=True, exist_ok=True)

    graph_path = graph_dir / "train_graph.svg"

    fig = plt.figure(figsize=(13, 6))
    grid = fig.add_gridspec(2, 2, height_ratios=[4, 1.4], wspace=0.22, hspace=0.28)

    ax_loss = fig.add_subplot(grid[0, 0])
    _plot_series(ax_loss, epochs, train_losses, "train loss")
    _plot_series(ax_loss, epochs, valid_losses, "valid loss")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(alpha=0.3)
    if ax_loss.lines:
        ax_loss.legend()

    ax_accuracy = fig.add_subplot(grid[0, 1])
    _plot_series(ax_accuracy, epochs, train_accuracies, "train accuracy")
    _plot_series(ax_accuracy, epochs, valid_accuracies, "valid accuracy")
    ax_accuracy.set_title("Accuracy")
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylabel("Accuracy")
    ax_accuracy.grid(alpha=0.3)
    if ax_accuracy.lines:
        ax_accuracy.legend()

    ax_loss_table = fig.add_subplot(grid[1, 0])
    _draw_summary_table(ax_loss_table, "Loss Summary", train_losses, valid_losses)

    ax_accuracy_table = fig.add_subplot(grid[1, 1])
    _draw_summary_table(ax_accuracy_table, "Accuracy Summary", train_accuracies, valid_accuracies)

    fig.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.92, wspace=0.22, hspace=0.34)
    plt.savefig(graph_path)
    if show:
        plt.show()
    plt.close()
