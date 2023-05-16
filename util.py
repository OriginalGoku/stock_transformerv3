from scipy.optimize import root
import numpy as np
import pandas as pd
from config import Config
from pathlib import Path
from typing import Tuple, List
import datetime


class Util:
    """
    The Util class provides various utility functions, such as shuffling data,
    reshaping data, normalizing data using z-score normalization,
    converting normalized data back to the original format,
    merging data dictionaries, generating sliding windows for train
    and test datasets, slicing and normalizing data, and loading and fixing data from files.
    """

    @staticmethod
    def shuffle_data(X_train: np.array, y_train: np.array):
        """
        Shuffle the training data and labels.

        Args:
            X_train (numpy.ndarray): Input training data.
            y_train (numpy.ndarray): Target training labels.

        Returns:
            tuple: Shuffled training data and labels.
        """
        idx = np.random.permutation(len(X_train))
        y_train = y_train[idx]
        X_train = X_train[idx]
        return X_train, y_train

    @staticmethod
    def reshape_data(X_train: np.array, X_test: np.array):
        """
        Reshape the input training and testing data.

        Args:
            X_train (numpy.ndarray): Input training data.
            X_test (numpy.ndarray): Input testing data.

        Returns:
            tuple: Reshaped training and testing data.
        """
        X_train = X_train.reshape(*X_train.shape, 1)
        X_test = X_test.reshape(*X_test.shape, 1)
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        return X_train, X_test

    @staticmethod
    def z_normalize(data, config: Config, mean_data=None, std_data=None):
        """
          Normalize the input data using z-score normalization.

          Args:
              data (numpy.ndarray): Input data to be normalized.
              mean_data (float, optional): Mean of the data. If None, the mean will be calculated from the data.
              std_data (float, optional): Standard deviation of the data. If None, the standard deviation will be calculated from the data.

          Returns:
              tuple: Z-score normalized data, mean value of the data, and standard deviation of the data.
        """
        if mean_data == None:
            mean_data = np.mean(data)

        if std_data == None:
            # Todo: We are changin the degree of freedom in std calculation to 1
            # this gives 1 degree of freedom to std
            std_data = np.std(data, ddof=config.std_degree_of_freedom)

        z_normal_data = (data - mean_data) / std_data
        return z_normal_data, mean_data, std_data

    @staticmethod
    def convert_to_original(normalized_window: np.array, config: Config, mean_data=None, std_data=None):
        """
        Convert the normalized window back to the original values based on the specified normalization method in the configuration.

        Args:
            normalized_window (numpy.ndarray): Normalized window to be converted. Shape of array is (batch_size X observation)
            config (Config): Configuration object containing the normalization method and other settings.
            mean_data (float or numpy.ndarray, optional): Mean value(s) used for normalization. If None, the mean value(s) will be retrieved from the configuration.
            std_data (float or numpy.ndarray, optional): Standard deviation value(s) used for normalization. If None, the standard deviation value(s) will be retrieved from the configuration.

        Returns:
            numpy.ndarray: Converted window with the original values.
        """
        if (config.normalizer == 'z_normalize'):
            print("normalized_window: ", len(normalized_window), "std_data: ", len(std_data), "mean_data: ",
                  len(mean_data))
            original_window = [(normalized_window[i] * std_data[i]) + mean_data[i] for i in
                               range(len(normalized_window))]

        elif config.normalizer == 'log':
            original_window = [np.exp(normalized_window[i]) for i in range(len(normalized_window))]
        elif config.normalizer == 'pct_change':
            pass
            # TO be implemented later
            # We would require the first value so to implement this, we need to
            # save the first value of each column we are normalizing using this method
            # # Calculate cumulative sum and add 1
            # cumulative_sum = np.cumsum(pct_change)
            # multiplier = cumulative_sum + 1

            # # Multiply multiplier with the first value
            # original_values = close_values[0] * multiplier

        # If no normalization is used, return the original window
        else:
            return normalized_window

        return np.array(original_window)

    @staticmethod
    def merge_data_dicts(data_dicts: List[dict]) -> dict:
        """
        Merges a list of data dictionaries into a single dictionary.
        This function is used to merge data generated by the
        sliding window generator from different files

        Args:
            data_dicts (List[dict]): A list of dictionaries containing data.

        Returns:
            dict: A merged dictionary containing the data from all input dictionaries.

        Raises:
            IndexError: If the input list is empty.
        """
        # Create an empty dictionary with keys from the first data dictionary
        merged_data_dict = {key: [] for key in data_dicts[0].keys()}

        # Iterate over each data dictionary
        for data_dict in data_dicts:
            # Iterate over the keys in the current data dictionary
            for key in data_dict.keys():
                # Merge the values of the current key into the merged dictionary
                merged_data_dict[key].extend(data_dict[key])

        # Return the merged dictionary
        return merged_data_dict

    @staticmethod
    def gen_sliding_windows(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates sliding windows of training and test data based on the provided configuration.

        Args:
            config (Config): An instance of the Config class containing the required parameters.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the merged training and test data as pandas DataFrames.

        Raises:
            None
        """
        # Initialize empty lists to store the data dictionaries for training and test sets

        train_data_dicts = []
        test_data_dicts = []

        # Iterate over each symbol file in the configuration's file list
        for symbol_file in config.file_list:
            print(f"Processing {symbol_file}")
            # Load the data from the symbol file using the Util class's load_file method
            data = Util.load_file(symbol_file, config)

            data = Util.fix_data(data, config)

            # The following complex code is only to comply with future versions of Python.
            # The simple version is:
            # data_train = data[data.index.date < config.training_cut_off_date]
            # data_test = data[data.index.date >= config.training_cut_off_date]
            # which generates a warning so in order to fix this issue, I had to write it in the following manner:

            # Separate the data into training and test sets based on the provided training cut-off date
            data_train = data[
                data.index.map(lambda x: x.date() if isinstance(x, datetime.datetime) else x) < pd.Timestamp(
                    config.training_cut_off_date).date()]
            data_test = data[
                data.index.map(lambda x: x.date() if isinstance(x, datetime.datetime) else x) >= pd.Timestamp(
                    config.training_cut_off_date).date()]

            # print("---------- gen_multiple_sliding_window --------------")
            # print(f"config.training_cut_off_date = {config.training_cut_off_date}")
            # print(f"data_train.iloc[0]: {data_train.iloc[0]}")
            # print(f"data_train.iloc[-1]: {data_train.iloc[-1]}")
            # print(f"data_test.iloc[0]: {data_test.iloc[0]}")
            # print(f"data_test.iloc[-1]: {data_test.iloc[-1]}")

            # print("---------- gen_multiple_sliding_window --------------")
            # Extract the symbol name from the symbol file path
            symbol_name = symbol_file.split('.')[0].split('_')[1]
            print(f"Generating Train Data for {symbol_name}...")
            train_data_dict = Util.slice_normalize_add_metadata_calculate_y(data_train, config)
            train_data_dict['symbol'] = [symbol_name] * len(train_data_dict['X'])
            train_data_dicts.append(train_data_dict)
            print(f"Generating Test Data for {symbol_name}...")
            test_data_dict = Util.slice_normalize_add_metadata_calculate_y(data_test, config)
            test_data_dict['symbol'] = [symbol_name] * len(test_data_dict['X'])
            test_data_dicts.append(test_data_dict)

        # Merge the data dictionaries into a single DataFrame and set the index as 'date'
        merged_train_data_dict = pd.DataFrame(Util.merge_data_dicts(train_data_dicts)).set_index('date')
        merged_test_data_dict = pd.DataFrame(Util.merge_data_dicts(test_data_dicts)).set_index('date')

        return merged_train_data_dict, merged_test_data_dict

    @staticmethod
    def slice_data(data: pd.DataFrame, config: Config):
        """
        Generates sliding windows for features (X) and target (y) based on the provided configuration.

        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame.
            config (Config): An instance of the Config class containing the required parameters.

        Returns:
            tuple: A tuple containing the generated sliding windows for X, y, high, low, close, open, and date.

        Raises:
            None
        """
        # Generate sliding windows for features (X) and target (y) using stride_tricks

        sliding_window_x = np.lib.stride_tricks.sliding_window_view(data[config.source],
                                                                    window_shape=(config.window_size,))
        sliding_window_y = np.lib.stride_tricks.sliding_window_view(data[config.source].iloc[config.window_size:],
                                                                    window_shape=(config.forecast_size,))
        # Generate sliding windows for high, low, and close values

        sliding_window_high = np.lib.stride_tricks.sliding_window_view(data['HIGH'].iloc[config.window_size:],
                                                                       window_shape=(config.forecast_size,))
        sliding_window_low = np.lib.stride_tricks.sliding_window_view(data['LOW'].iloc[config.window_size:],
                                                                      window_shape=(config.forecast_size,))
        sliding_window_close = np.lib.stride_tricks.sliding_window_view(data['CLOSE'].iloc[config.window_size:],
                                                                        window_shape=(config.forecast_size,))
        sliding_window_open = np.lib.stride_tricks.sliding_window_view(data['OPEN'].iloc[config.window_size:],
                                                                       window_shape=(config.forecast_size,))
        # the date sliding window starts on the day we know the open value
        sliding_window_date = np.lib.stride_tricks.sliding_window_view(data.index[config.window_size:],
                                                                       window_shape=(config.forecast_size,))

        print(f"len(sliding_window_x) = {len(sliding_window_x)} -- len(sliding_window_y) {len(sliding_window_y)}")
        # Original Code:
        # sliding_window_x_including_open = [np.append(sliding_window_x[i], sliding_window_open[i][0]) for i in range(len(sliding_window_open))]
        # Optimal code:
        sliding_window_x_including_open = [np.append(x, o[0]) for x, o in zip(sliding_window_x, sliding_window_open)]

        return sliding_window_x, sliding_window_x_including_open, sliding_window_y, sliding_window_high, sliding_window_low, sliding_window_close, sliding_window_open, sliding_window_date

    @staticmethod
    def slice_normalize_add_metadata_calculate_y(data: pd.DataFrame, config: Config) -> dict:
        """
        Slices the data, normalizes the windows, adds metadata (std and later mean...), and calculates the target values based on the provided configuration value for y_function.

        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame. this dataframe must contain columns 'OPEN', 'HIGH', 'LOW', 'CLOSE' and source column (for example if OHLC is used then the dataframe must contain OHLC)
            config (Config): An instance of the Config class containing the required parameters.

        Returns:
            dict: A dictionary containing the processed data, including sliding windows, normalized windows, metadata, and target values.

        Raises:
            None
        """

        def normalize_windows(window_x_including_open: List[np.ndarray], sliding_window_y: List[np.ndarray],
                              config: Config) -> Tuple[
            List[np.ndarray], List[float], List[float], List[float], List[float], List[float], List[float]]:
            """
            Normalizes the sliding windows and calculates mean and standard deviation.

            Args:
                window_x_including_open (List[np.ndarray]): Sliding windows for features (X) including the open value.
                sliding_window_y (List[np.ndarray]): Sliding windows for the target (y).
                config (Config): An instance of the Config class containing the required parameters.

            Returns:
                Tuple[List[np.ndarray], List[float], List[float], List[float], List[float], List[float]]: A tuple containing the normalized windows for X, mean values for X, standard deviation values for X,
                normalized windows for y, mean values for y, standard deviation values for y.

            Raises:
                None
            """
            normalized_windows_x, mean_x_list, std_x_list, normalized_windows_y, mean_y_list, std_y_list, y_original = [], [], [], [], [], [], []

            for wx, wy in zip(window_x_including_open, sliding_window_y):

                if config.normalizer == 'z_normalize':
                    norm_wx, mean_x, std_x = Util.z_normalize(wx, config)

                    if config.y_function == 'last':
                        y_value = wy[-1]
                    else:
                        y_value = eval(f"np.{config.y_function}(wy)")

                    # mean_y, std_y = np.mean(np.concatenate((wx[:-1], wy))), np.std(np.concatenate((wx[:-1], wy)), ddof=config.std_degree_of_freedom)
                    mean_y, std_y = np.mean(np.append(wx[:-1], y_value)), np.std(np.append(wx[:-1], y_value),
                                                                                 ddof=config.std_degree_of_freedom)
                    norm_wy, mean_wy, std_wy = Util.z_normalize(y_value, config, mean_y, std_y)

                    if config.include_std:
                        log_std = np.log(std_x)
                        norm_wx = np.insert(norm_wx, 0, log_std)

                mean_x_list.append(mean_x)
                std_x_list.append(std_x)
                mean_y_list.append(mean_y)
                std_y_list.append(std_y)
                normalized_windows_x.append(norm_wx)
                normalized_windows_y.append(norm_wy)
                y_original.append(y_value)

            return normalized_windows_x, mean_x_list, std_x_list, normalized_windows_y, mean_y_list, std_y_list, y_original

        print(f"Generating Sliding Window on {config.source} - (window_size = {config.window_size}, "
              f"normalizer = {config.normalizer}) - y_function = {config.y_function}"
              f" - len(data[{config.source}]) = {len(data)}")
        print(f"First Element of data source: {data.iloc[0].name}")

        windows = Util.slice_data(data, config)
        sliding_window_x, window_x_including_open, sliding_window_y, sliding_window_high, sliding_window_low, sliding_window_close, sliding_window_open, sliding_window_date = windows

        # todo: this code removes open from last element of X
        # normalized_window_x, mean_x_list, std_x_list, normalize_windows_y, mean_y_list, std_y_list, y_original = normalize_windows(window_x_including_open, sliding_window_y, config)
        normalized_window_x, mean_x_list, std_x_list, normalize_windows_y, mean_y_list, std_y_list, y_original = normalize_windows(
            sliding_window_x[:len(sliding_window_y)], sliding_window_y, config)

        data_dict = {
            'date': [d[0] for d in sliding_window_date],
            'open': [o[0] for o in sliding_window_open],
            'high': [np.max(h) for h in sliding_window_high],
            'low': [np.min(l) for l in sliding_window_low],
            'close': [c[-1] for c in sliding_window_close],
            'X': normalized_window_x,
            'y': normalize_windows_y,
            # Todo: Keepsliding_window_x[len(normalized_window_x):] for future predictions
            'X_original': sliding_window_x[:len(sliding_window_y)],
            'y_original': y_original,
            'mean_x': mean_x_list,
            'std_x': std_x_list,
            'mean_y': mean_y_list,
            'std_y': std_y_list,
            'atr': data['ATR'][:len(normalized_window_x)]
        }

        print(f"Final Len X: {len(data_dict['X'])}")
        return data_dict

    @staticmethod
    def load_file(file_name: str, config: Config, folder_path: Path = None):
        """
        Loads a CSV file into a pandas DataFrame.

        Args:
            file_name (str): The name of the file to load.
            config (Config): An instance of the Config class containing the required parameters.
            folder_path (Path, optional): The folder path where the file is located. Defaults to None.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            None
        """
        if folder_path is None:
            file_path = config.data_folder / file_name
        else:
            file_path = folder_path / file_name

        print(f"Loading file: {file_path}")
        data = pd.read_csv(file_path, index_col=[0], parse_dates=True).rename_axis('Date')
        # Todo if we want to use shorter time frames, we need to change this part
        # Convert the index to date only
        data.index = pd.Index([dt.date() for dt in data.index])

        return data

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int):
        high = data['HIGH']
        low = data['LOW']
        close = data['CLOSE']

        # Calculate the True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate the Average True Range (ATR)
        atr = tr.rolling(period).mean()

        return atr

    @staticmethod
    def fix_data(data, config: Config):
        """
        Fixes the data by performing various transformations and modifications.

        Args:
            data (pd.DataFrame): The input data as a pandas DataFrame.
            config (Config): An instance of the Config class containing the required parameters.

        Returns:
            pd.DataFrame: The fixed data as a pandas DataFrame.

        Raises:
            None
        """

        # Select the columns specified in 'usable_data_col' from the configuration
        data = data.loc[:, config.usable_data_col]

        # Convert column names to uppercase
        data.columns = [x.upper() for x in data.columns]

        # Calculate the percentage change of 'CLOSE'
        data['PCT_CHANGE'] = data['CLOSE'].pct_change()

        # Calculate the rolling mean of 'CLOSE'
        data['CLOSE_MA'] = data['CLOSE'].rolling(config.ma_len).mean()

        # Calculate additional columns based on specific formulas
        data['HLCC'] = (data['HIGH'] + data['LOW'] + data['CLOSE'] + data['CLOSE']) / 4
        data['OHLC'] = (data['OPEN'] + data['HIGH'] + data['LOW'] + data['CLOSE']) / 4
        data['DETREND'] = data['CLOSE'] - data['CLOSE_MA']
        data['LOG_DETREND'] = np.log(data['CLOSE']) - np.log(data['CLOSE_MA'])
        data['ATR'] = Util.calculate_atr(data, config.atr_period)

        if config.use_quantile_filter:
            # Clip 'LOG_DETREND' values based on quantile filter
            quantile_filter = config.quantile_filter
            data['LOG_DETREND'] = data['LOG_DETREND'].clip(
                lower=data['LOG_DETREND'].quantile(1 - quantile_filter),
                upper=data['LOG_DETREND'].quantile(quantile_filter))

        # Drop rows with missing values
        data.dropna(inplace=True)

        print(f"Loaded {len(data)} rows")
        print(f"First element of Loaded data {data.index[0]}")
        print(f"Last element of Loaded data {data.index[-1]}")

        return data

    @staticmethod
    def save_csv(data, file_name):
        """
        Saves the data to a CSV file.

        Args:
            data (pd.DataFrame): The data to be saved as a pandas DataFrame.
            file_name (str): The name of the CSV file.

        Returns:
            None

        Raises:
            None
        """

        # Save the data to the specified CSV file
        data.to_csv(file_name, index=None)

    @staticmethod
    def find_original_y(X, y):
        def equation(Unknown_Variable):
            combined_data = np.append(X, Unknown_Variable)
            return (Unknown_Variable - np.mean(combined_data)) / np.std(combined_data) - y

        Unknown_Variable_solution = root(equation, [1])  # Initial guess is 1
        # print('Unknown_Variable:', Unknown_Variable_solution.x[0])
        return Unknown_Variable_solution.x[0]

    @staticmethod
    def generate_data(config: Config):
        train_data, test_data = Util.gen_sliding_windows(config)

        X_train = np.array(train_data['X'].to_list())
        X_test = np.array(test_data['X'].to_list())
        y_train = np.array(train_data['y'].to_list())
        y_test = np.array(test_data['y'].to_list())

        X_train, y_train = Util.shuffle_data(X_train, y_train)
        X_train, X_test = Util.reshape_data(X_train, X_test)

        # Plotter.plot_hist_distribution(y_train, y_test, "y_train", 'y_test', config.result_folder)

        return X_train, X_test, y_train, y_test, train_data, test_data

    # myutil = Util()
# train_data, test_data = myutil.gen_sliding_windows(global_config)
