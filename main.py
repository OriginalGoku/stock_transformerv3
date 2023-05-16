from plots import Plotter
from transformer import Transformer
from util import Util
from mlp import MLP
from config import Config
import pandas as pd
import numpy as np
from analyzer import analyze_results


def train_model(X_train, y_train, config, model_name):
    print(f"Training {model_name} ... ")
    if model_name == 'transformer':
        # history, model = Transformer().construct_transformer(X_train=X_train, y_train=y_train, window_size=config.window_size,
        #                                                  forecast_size=config.forecast_size, source=config.source,
        #                                                  data_folder=config.data_folder, normalizer=config.normalizer,
        #                                                  model_folder=config.models_folder, **config.transformer_setting)

        history, model = Transformer().train_model(config, 'AdaptviePooling', X_train, y_train)
    else:
        mlp_model = MLP()
        raw_model = mlp_model.build_model(X_train, model_name)
        history, model = mlp_model.train_model(config, model_name, raw_model, X_train, y_train)

    # transformer.evaluate_model(model, X_test, y_test)
    return model


def generate_results(global_config):
    model_names = ['conv1D_model', 'gru_model', 'lstm_model', 'transformer']
    model_list = []
    y_pred_list = []
    y_pred_from_model = []
    for model_name in model_names:
        if model_name != 'transformer':
            global_config.transformer_setting['epoc'] = 30
        else:
            global_config.transformer_setting['epoc'] = 3

        model = train_model(X_train, y_train, global_config, model_name)
        y_pred_normal = model.predict(X_test)
        model_list.append(model)
        y_pred_original = [Util.find_original_y(X, y[0]) for X, y in
                           zip(test_data['X_original'].to_list(), y_pred_normal)]
        y_pred_from_model.append(y_pred_normal)
        # y_pred_original = (y_pred.flatten()*test_data['std_y'].to_list())+test_data['mean_y'].to_list()
        y_pred_list.append(y_pred_original)
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
        print(f"Results for {model_name}")
        print("Test Loss:", test_loss)
        print("Test Mean Absolute Error:", test_mae)
        print("--------------------------")

    return y_pred_list, y_pred_from_model, model_names


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    global_config = Config(y_function="mean", normalizer='z_normalize',
                           source='HLCC', )
    X_train, X_test, y_train, y_test, train_data, test_data = Util.generate_data(global_config)
    Plotter.plot_hist_distribution(y_train, y_test, "y_train", "y_test", global_config)

    y_pred, y_pred_norm, model_names = generate_results(global_config)

    test_data['mean_prediction_'+global_config.source] = np.array(pd.DataFrame(y_pred).T.mean(axis=1))
    test_data['trade_result_mean'] = analyze_results(test_data, test_data['mean_prediction_'+global_config.source], global_config)
    for idx, model_name in enumerate(model_names):
        col_name = 'y_pred_' + global_config.source + "_" + model_name
        test_data[col_name] = y_pred[idx]
        # test_data['log' + col_name + '_to_open'] = np.log(test_data[col_name] / test_data['open'])
        test_data['trade_result_'+model_name] = analyze_results(test_data, y_pred[idx], global_config)



    test_data.to_csv('results.csv', index='date')




