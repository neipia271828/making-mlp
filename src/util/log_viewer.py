import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from CONSTANTS import CONSTANTS

df = pd.read_csv(f"data/{CONSTANTS.PROJECT}/logs/{CONSTANTS.PROJECT}.csv")


def _get_first_available_column(frame: pd.DataFrame, *column_names: str) -> pd.Series:
    for column_name in column_names:
        if column_name in frame.columns:
            return frame[column_name]
    raise KeyError(f"None of the columns exist: {', '.join(column_names)}")


def _get_optional_column(frame: pd.DataFrame, *column_names: str) -> pd.Series | None:
    for column_name in column_names:
        if column_name in frame.columns:
            return frame[column_name]
    return None


def _annotate_points(
    ax,
    x_values,
    y_values,
    fmt: str = ".3f",
    xytext: tuple[int, int] = (0, 6),
    va: str = "bottom",
) -> None:
    for x_value, y_value in zip(x_values, y_values):
        ax.annotate(
            format(y_value, fmt),
            (x_value, y_value),
            textcoords="offset points",
            xytext=xytext,
            ha="center",
            va=va,
            fontsize=8,
        )


def main():
    x = df["time_stamp"]
    train_accuracy = _get_optional_column(df, "train_accuracy_last10_avg", "train_accuracy")
    valid_accuracy = _get_first_available_column(df, "valid_accuracy_last10_avg", "valid_accuracy", "accuracy")
    train_loss = _get_first_available_column(df, "train_loss_last10_avg", "train_loss", "total_loss")
    valid_loss = _get_optional_column(df, "valid_loss_last10_avg", "valid_loss")
    req_time = df["req_time"]
    ep_time = df["ep_time"]
    num_epochs = df["NUM_EPOCHS"]
    learning_rate = df["L_LATE"]
    batch_size = df["BATCHSIZE"]

    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    # 1. Evaluation metrics
    eval_ax = axes[0]
    eval_ax.plot(x, valid_accuracy, marker="o", label="valid accuracy (last10 avg)", color="tab:cyan")
    _annotate_points(eval_ax, x, valid_accuracy)
    if train_accuracy is not None:
        eval_ax.plot(x, train_accuracy, marker="o", label="train accuracy (last10 avg)", color="tab:blue")
        _annotate_points(eval_ax, x, train_accuracy)
    eval_ax.plot(x, train_loss, marker="s", label="train loss (last10 avg)", color="tab:red")
    _annotate_points(eval_ax, x, train_loss, xytext=(0, 8), va="bottom")
    if valid_loss is not None:
        eval_ax.plot(x, valid_loss, marker="s", label="valid loss (last10 avg)", color="tab:orange")
        _annotate_points(eval_ax, x, valid_loss, xytext=(0, -10), va="top")
    eval_ax.set_title("Evaluation Metrics")
    eval_ax.set_ylabel("Accuracy / Loss")
    eval_ax.legend(loc="best")

    # 2. Runtime performance
    perf_ax = axes[1]
    perf_time_ax = perf_ax.twinx()
    perf_ax.plot(x, req_time, marker="o", label="total time", color="tab:green")
    _annotate_points(perf_ax, x, req_time, ".2f")
    perf_time_ax.plot(x, ep_time, marker="o", label="time per epoch", color="tab:olive")
    _annotate_points(perf_time_ax, x, ep_time, ".2f")
    perf_ax.set_title("Runtime Performance")
    perf_ax.set_ylabel("Total Seconds")
    perf_time_ax.set_ylabel("Seconds per Epoch")
    handles1, labels1 = perf_ax.get_legend_handles_labels()
    handles2, labels2 = perf_time_ax.get_legend_handles_labels()
    perf_ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

    # 3. Hyperparameters
    hyper_ax = axes[2]
    hyper_lr_ax = hyper_ax.twinx()
    hyper_ax.plot(x, num_epochs, marker="o", label="epochs", color="tab:purple")
    _annotate_points(hyper_ax, x, num_epochs, ".0f")
    hyper_ax.plot(x, batch_size, marker="o", label="batch size", color="tab:brown")
    _annotate_points(hyper_ax, x, batch_size, ".0f")
    hyper_lr_ax.plot(x, learning_rate, marker="o", label="learning rate", color="tab:pink")
    _annotate_points(hyper_lr_ax, x, learning_rate, ".4f")
    hyper_ax.set_title("Hyperparameters")
    hyper_ax.set_ylabel("Epochs / Batch Size")
    hyper_lr_ax.set_ylabel("Learning Rate")
    handles1, labels1 = hyper_ax.get_legend_handles_labels()
    handles2, labels2 = hyper_lr_ax.get_legend_handles_labels()
    hyper_ax.legend(handles1 + handles2, labels1 + labels2, loc="best")

    axes[2].set_xlabel("time_stamp")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        ax.margins(x=0.05, y=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
