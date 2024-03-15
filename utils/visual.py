from rich.table import Table
from rich.console import Console
import plotext as plt

STYLE_COLOR = {"train": "blue", "valid": "green", "test": "red"}


def show_table(history):
    # Extract metrics and modes
    metrics = history["test"].keys()
    modes = history.keys()

    # Show table
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Epoch")
    for mode in modes:
        for metric in metrics:
            table.add_column(
                f"{mode.capitalize()} {metric.upper()}", style=STYLE_COLOR[mode]
            )
    for epoch in range(len(history["test"]["loss"])):
        row = [str(epoch + 1)]
        for mode in modes:
            for metric in metrics:
                value = history[mode][metric][epoch]
                row.append(f"{value:.3f}")
        table.add_row(*row)
    console.print(table)


def show_plot(history):
    # Extract metrics and modes
    metrics = history["test"].keys()
    modes = history.keys()

    # Show plot for each metric
    plt.clf()  # Clear any previous plot data
    plt.plot_size(200, 20)
    plt.subplots(1, len(metrics))
    for i, metric in enumerate(metrics):
        epochs = [epoch + 1 for epoch in range(len(history["test"]["loss"]))]
        plt.subplot(1, i + 1)

        # Plotting's settings
        plt.title(f"{metric.upper()} vs Epoch")
        plt.xticks(epochs)

        # Plot data
        for mode in modes:
            plt.plot(
                epochs,
                history[mode][metric],
                color=STYLE_COLOR[mode],
                label=f"{mode.capitalize()} {metric.upper()}",
            )

    # Show all plots
    plt.show()
