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

    trade_results = []
    for _, row in result_df.iterrows():
        y = y_pred.loc[_].values[0]

        # Taking short trades
        if (row['open'] - (row['atr'] / 2)) > y:  # row['y_pred']
            # y_pred is reached
            if row['low'] <= y:  # row['y_pred']:
                # result_df.loc[_, 'trade_result'] = np.log(row['open'] / row['y_pred'])
                trade_results.append(np.log(row['open'] / y))  # row['y_pred']))
            # y_pred is not reached
            else:
                # result_df.loc[_, 'trade_result'] = np.log(row['open'] / row['close'])
                trade_results.append(np.log(row['open'] / row['close']))

        # Taking Long Trades
        elif (row['open'] + (row['atr'] / 2)) < y:  # row['y_pred']:
            # y_pred is reached
            if row['high'] >= y:  # row['y_pred']:
                # result_df.loc[_, 'trade_result'] = np.log(
                #     y / row['open'])  # np.log(row['y_pred'] / row['open'])
                trade_results.append(np.log(y / row['open']))

            # y_pred is not reached
            else:
                # result_df.loc[_, 'trade_result'] = np.log(row['close'] / row['open'])
                trade_results.append(np.log(row['close'] / row['open']))
        else:
            # No trade was taken
            trade_results.append(0.0)

    # return result_df
    return trade_results

