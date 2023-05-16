from util import Util
import numpy as np
import pandas as pd
from config import Config


def analyze_results(result_df: pd.DataFrame, y_pred, config: Config):
    y_pred=pd.DataFrame(y_pred)
    y_pred.index = result_df.index

    # result_df['y_pred'] = [Util.find_original_y(X, y[0]) for X, y in
    #                        zip(result_df['X_original'].to_list(), y_pred_normalized)]
    # result_df['y_pred'] = y_pred
    # result_df['log_pred_open'] = np.log(result_df['y_pred'] / result_df['open'])
    # result_df['diff_pred_open'] = result_df['y_pred'] - result_df['open']
    # if config.source == 'HIGH':
    #     if config.entry_condition == 'atr':
    #         entry_condition = result_df[result_df['diff_pred_open'] >= result_df['atr']]
    #     elif config.entry_condition == 'threshold':
    #         entry_condition = result_df[result_df['log_pred_open'] >= config.threshold]
    #
    #     entry_condition['target_reached'] = entry_condition['y_pred'] <= entry_condition['high']
    #
    #     entry_condition = entry_condition.assign(
    #         trade_results=[
    #             np.log(row['close'] / row['open'])
    #             if (row['y_pred'] > row['high'])
    #             else np.log(row['y_pred'] / row['open'])
    #             for _, row in entry_condition.iterrows()
    #         ])
    # elif config.source == 'LOW':
    #     if config.entry_condition == 'atr':
    #         entry_condition = result_df[result_df['diff_pred_open'] <= -result_df['atr']]
    #     elif config.entry_condition == 'threshold':
    #         entry_condition = result_df[result_df['log_pred_open'] <= -config.threshold]
    #
    #     entry_condition['target_reached'] = entry_condition['y_pred'] >= entry_condition['low']
    #
    #     entry_condition = entry_condition.assign(
    #         trade_results=[
    #             np.log(row['open'] / row['close'])
    #             if (row['y_pred'] < row['low'])
    #             else np.log(row['open'] / row['y_pred'])
    #             for _, row in entry_condition.iterrows()
    #         ]
    #     )
    # else:

    # trade_results = []
    # for _, row in result_df.iterrows():
    #     y = y_pred.loc[_].values[0]
    #
    #     # Taking short trades
    #     if (row['open'] - (row['atr'] / 2)) > y:
    #         # y_pred is reached
    #         if row['low'] <= y:
    #             result = trade_results.append(np.log(row['open'] / y))
    #
    #         # y_pred is not reached
    #         else:
    #             result = np.log(row['open'] / row['close'])
    #
    #     # Taking Long Trades
    #     elif (row['open'] + (row['atr'] / 2)) < y:
    #         # y_pred is reached
    #         if row['high'] >= y:
    #             result = np.log(y / row['open'])
    #
    #         # y_pred is not reached
    #         else:
    #             result = np.log(row['close'] / row['open'])
    #     else:
    #         result = 0.0
    #         # No trade was taken
    #     if type(result) != 'int':
    #         result = result[0]
    #     trade_results.append(result)

    def take_short_trade(row, y):
        return np.log(row['open'] / y) if row['low'] <= y else np.log(row['open'] / row['close'])

    def take_long_trade(row, y):
        return np.log(y / row['open']) if row['high'] >= y else np.log(row['close'] / row['open'])

    trade_results = [
        take_short_trade(row, y_pred.loc[_].values[0]) if (row['open'] - (row['atr'] / 2)) > y_pred.loc[_].values[0]
        else take_long_trade(row, y_pred.loc[_].values[0]) if (row['open'] + (row['atr'] / 2)) < y_pred.loc[_].values[0]
        else 0.0
        for _, row in result_df.iterrows()
    ]

    # return result_df
    return trade_results

