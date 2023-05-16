import warnings
from config import Config
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from typing import List, Optional, Tuple
from pathlib import Path

from pandas import DatetimeTZDtype
from pandas.tseries.offsets import BDay
import mplfinance as mpf


class Plotter:
    """
    Plotter class is used to visualize various aspects of a model's performance
    and the distribution of data. It creates plots for the training and validation
    loss, history metrics, scatter plots of true vs predicted values, and the histogram
    of the differences between true and predicted values.
    """

    def __init__(self, y_test, config: Config, history: keras.callbacks.History,
                 save_results=True, display_plots=True):
        self.config = config
        self.history = history
        self.y_test = y_test
        self.save_results = save_results
        self.display_plots = display_plots
        self.fig_size = config.fig_size

    @staticmethod
    def plot_hist_distribution(first_input, second_input, first_input_title, second_input_title, config: Config,
                               bins=100):
        """
        Plots the histogram for the train and test set y values.

        :param first_input: First input data (usually train set)
        :param second_input: Second input data (usually test set)
        :param first_input_title: Title for the first input data
        :param second_input_title: Title for the second input data
        :param config: Configuration object
        :param bins: Number of bins for the histograms
        """

        hist_train, bins_train = np.histogram(first_input, bins=bins)
        hist_test, bins_test = np.histogram(second_input, bins=bins)

        plt.bar(bins_train[:-1], hist_train, width=(bins_train[1] - bins_train[0]), label=first_input_title)
        plt.bar(bins_test[:-1], hist_test, width=(bins_test[1] - bins_test[0]), label=second_input_title)
        plt.legend()

        file_name = Path("Train and Test y Distribution.png")
        plt.savefig(config.plot_folder / file_name)
        plt.show()
        plt.close()

    # Model Evaluation plots
    def plot_train_validation_loss(self):
        # Plot the training and validation loss
        plt.figure(figsize=self.fig_size)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Improvement of the Network')
        plt.legend()
        if self.save_results:
            plt.savefig(Path(self.result_folder / 'Network train and validation loss.png'))
        plt.show()

    def plot_history_metrics(self):
        total_plots = len(self.history.history)
        cols = total_plots // 2

        rows = total_plots // cols

        if total_plots % cols != 0:
            rows += 1

        pos = range(1, total_plots + 1)
        plt.figure(figsize=self.fig_size)
        for i, (key, value) in enumerate(self.history.history.items()):
            plt.subplot(rows, cols, pos[i])
            plt.plot(range(len(value)), value)
            plt.title(str(key))
        if self.save_results:
            plt.savefig(Path(self.result_folder / 'Network history metrics.png'))
        plt.show()

    def plot_scatter_true_vs_predicted(self, y_pred, start_: int, end_: int, plot_title: Optional[str] = None):
        fig = plt.figure(figsize=self.fig_size)
        print(f"Plotting from {start_} to {end_} for y_test = {len(self.y_test)}, predictions = {len(y_pred)} ")
        # Plot the limited range of true values vs the predicted values
        plt.scatter(np.arange(start_, end_), y_pred[start_:end_], alpha=0.5, marker='x', color='red',
                    label='Predicted')
        plt.scatter(np.arange(start_, end_), self.y_test.reshape(-1, 1)[start_:end_], alpha=0.5, marker='o',
                    color='blue',
                    label='True')
        plt.ylabel("Predicted/True Values")
        plt.title(f"{plot_title} True Values vs Predicted Values")
        plt.legend()

        file_name = Path(plot_title + ' Scatter True vs Predict' + self.config.file_name_format + ".png")

        if self.save_results:
            plt.savefig(self.config.result_folder / file_name)

        if self.display_plots:
            plt.show()

        plt.close()

    # def plot_histogram_y_test_minus_y_pred(self, y_pred: np.array, plot_title: Optional[str] =None, bins=30, clip_value = 1):
    #     # Calculate the differences between true and predicted values
    #     differences = (self.y_test - y_pred.reshape(-1, )).flatten()
    #     differences_pct = np.round([np.log(y_pred.reshape(-1, )[i] / self.y_test[i])
    #                                 for i in range(len(self.y_test))], 4).flatten()

    #     if self.config.source=="HIGH":
    #       winning_pct = len(differences_pct[differences_pct>=0.0])/len(differences_pct)
    #     elif self.config.source =='LOW':
    #       winning_pct = len(differences_pct[differences_pct<=0.0])/len(differences_pct)

    #     print(f"Winning Pct {round(100*winning_pct,2)}%")
    #     # Plot the histogram of differences
    #     # Clip differences larger than 100%
    #     differences_pct = np.clip(differences_pct, -clip_value, clip_value)
    #     if (np.max(np.abs(differences_pct))>clip_value):
    #         warnings.warn(f"plot_histogram_y_test_minus_y_pred -> Difference between prediction and true values exceeds +-{100*clip_value}%. data is clipped")

    #     plt.hist(differences_pct, bins=bins, color='purple')
    #     plt.xlabel("Difference")
    #     plt.ylabel("Frequency")
    #     plt.title(f"Histogram of Differences between True and Predicted Values for {plot_title}")
    #     file_name = plot_title + ' Histogram True-Predict' + self.config.file_name_format + ".png"
    #     if self.save_results:
    #         plt.savefig(self.config.result_folder / file_name)
    #     if self.display_plots:
    #       plt.show()

    #     plt.close()

    import matplotlib.pyplot as plt

    def plot_histogram_y_test_minus_y_pred(self, y_pred: np.array, plot_title: Optional[str] = None, bins: int = 30,
                                           clip_value: float = 1, threshold: float = 0.00125) -> None:
        # Calculate the differences between true and predicted values
        # differences = (self.y_test/y_pred.reshape(-1, )).flatten()
        print(
            f'len(self.y_test[self.y_test>y_pred]) = {len(self.y_test[self.y_test > y_pred])} / {len(y_pred)} = {len(self.y_test[self.y_test > y_pred]) / len(y_pred)}')
        differences_pct = np.round([np.log(self.y_test[i] / y_pred.reshape(-1, )[i])
                                    for i in range(len(self.y_test))], 4).flatten()

        if self.config.source == "HIGH":
            won_trades = differences_pct[differences_pct >= -threshold]
            lost_trades = differences_pct[differences_pct < -threshold]
        elif self.config.source == 'LOW':
            won_trades = differences_pct[differences_pct <= threshold]
            lost_trades = differences_pct[differences_pct > threshold]

        winning_pct = len(won_trades) / len(differences_pct)

        print(f"Winning Pct {round(100 * winning_pct, 2)}%")
        # Plot the histogram of differences
        # Clip differences larger than 100%
        differences_pct = np.clip(differences_pct, -clip_value, clip_value)
        if np.max(np.abs(differences_pct)) > clip_value:
            warnings.warn(
                f"plot_histogram_y_test_minus_y_pred -> Difference between prediction and true values exceeds +-{100 * clip_value}%. data is clipped")

        positive_color = 'green'  # color for values above 0
        negative_color = 'red'  # color for values below 0

        plt.hist(won_trades, bins=bins, color=positive_color, label='Won Trades')
        plt.hist(lost_trades, bins=bins, color=negative_color, label='Lost Trades')

        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.title(f"Histogram of Differences between True and Predicted Values for {plot_title}")

        # Add a legend showing the percentage of data above 0 and below 0
        above_zero_pct = round(100 * winning_pct, 2)
        below_zero_pct = round(100 * (1 - winning_pct), 2)

        legend_text = f"Winning: {above_zero_pct}%\nLosing: {below_zero_pct}%"
        custom_legend = [plt.Line2D([0], [0], color='w', lw=4, label=legend_text)]
        plt.legend(handles=custom_legend, title='Trade Performance')

        # legend_text = f"Winning: {above_zero_pct}%\nLosing: {below_zero_pct}%"
        # plt.legend(title='Trade Performance', label=legend_text)

        file_name = plot_title + ' Histogram True-Predict' + self.config.file_name_format + ".png"
        if self.save_results:
            plt.savefig(self.config.result_folder / file_name)
        if self.display_plots:
            plt.show()

        plt.close()

