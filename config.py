# This function defines a data class called "Config" that represents the core
# data structure used in an AI-based stock prediction system.
# The class contains various attributes that configure the behavior of the system.
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import pandas as pd

@dataclass
class Config:
    file_list: List[str] = field(default_factory=lambda: ['BATS_DIA.csv', 'BATS_IWM.csv', 'BATS_TLT.csv', 'BATS_XLB.csv', 'BATS_XLF.csv', 'BATS_XLK.csv', 'BATS_XLP.csv', 'BATS_XLU.csv', 'BATS_XLV.csv', 'BATS_XLY.csv', 'BATS_SPY.csv','BATS_QQQ.csv', 'BATS_XLE.csv'])
    # normalizer: str = field(default='z_normalize', metadata={'allowed': ['z_normalize', 'pct_change', 'log', 'none']})
    normalizer: str = field(default='z_normalize', metadata={
        'allowed': ['z_normalize', 'log']})  # Did not implement these yet, 'pct_change', 'log', 'none']})
    use_quantile_filter: bool = False
    include_std: bool = True
    std_degree_of_freedom = 0
    # This value determines how y_values are calculated.
    # mean = np.mean(forecast_size bars after window_size)
    # min_max = use min and max for y. this variable is only accepted if the values of source = ['HIGH', 'LOW']
    # last = value of source at [window_size+forecast_size]
    y_function: str = field(default='mean', metadata={'allowed': ['mean', 'min', 'max', 'last']})
    window_size: int = 10
    forecast_size: int = 3
    quantile_filter: float = 0.99

    rounding_precision = 4
    ma_len = 5
    atr_period: int = 14

    # Plotting:
    fig_size = (15, 5)

    # Trade Settings
    source: str = field(default='HIGH', metadata={'allowed': ['HIGH', 'LOW', 'CLOSE', 'OHLC', 'HLCC']})
    entry_condition: str = field(default='atr', metadata={'allowed': ['atr', 'threshold']})

    usable_data_col: List[str] = field(default_factory=lambda: ['close', 'open', 'high', 'low', 'Volume'])

    data_folder: Path = Path('data/daily')
    result_folder: Path = Path('results')
    models_folder: Path = Path('models')
    plot_folder: Path = Path('plots')

    training_cut_off_date: pd.Timestamp = pd.to_datetime('2018-01-03')
    # The original time format of stock data contains timezone information such as: 09:30:00-05:00')

    # Trade Analyzer:
    # Distance between open and predicted value for entering a trade
    threshold: float = 0.01
    optimizer: str = field(default='RMSprop', metadata={
        'allowed': ['Adam', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl']})

    # transformer_setting = {'epoc': 1, 'optimizer_choice': 'adamax', 'num_heads': 8, 'head_size': 256, 'ff_dim': 6,
    transformer_setting = {'epoc': 1, 'optimizer_choice': optimizer, 'num_heads': 8, 'head_size': 256, 'ff_dim': 6,
                           'num_transformer_blocks': 6, 'mlp_units': 512, 'dropout': 0.5, 'mlp_dropout': 0.6,
                           'learning_rate': 0.00134, 'min_learning_rate': 0.00001, 'validation_split': 0.2,
                           'batch_size': 32, 'loss': 'mean_squared_error', 'metrics': 'mean_absolute_error'}

    @property
    def file_name_format(self):
        return f"Window {self.window_size} - Forecast {self.forecast_size} - Source {self.source} - {self.normalizer}"

    # Check if the folders exist, if not, create them after instantiating the class
    def __post_init__(self):
        # Check if the normalizer choice is valid
        allowed_normalizers = self.__dataclass_fields__['normalizer'].metadata['allowed']
        allowed_y_functions = self.__dataclass_fields__['y_function'].metadata['allowed']
        allowed_sources = self.__dataclass_fields__['source'].metadata['allowed']
        allowed_optimizers = self.__dataclass_fields__['optimizer'].metadata['allowed']
        allowed_entry_conditions = self.__dataclass_fields__['entry_condition'].metadata['allowed']
        if self.normalizer not in allowed_normalizers:
            raise ValueError(
                f"Invalid normalizer choice '{self.normalizer}', allowed choices are {allowed_normalizers}")
        if self.y_function not in allowed_y_functions:
            raise ValueError(
                f"Invalid y_function choice '{self.y_function}', allowed choices are {allowed_y_functions}")
        if self.source not in allowed_sources:
            raise ValueError(f"Invalid source choice '{self.source}', allowed choices are {allowed_sources}")
        if self.optimizer not in allowed_optimizers:
            raise ValueError(f"Invalid optimizer choice '{self.optimizer}', allowed choices are {allowed_optimizers}")

        if self.entry_condition not in allowed_entry_conditions:
            raise ValueError(
                f"Invalid entry condition {self.entry_condition}, allowed choises are {allowed_entry_conditions}")

        if (
                ((self.y_function == 'min') and (self.source == 'HIGH')) or
                ((self.y_function == 'max') and (self.source == 'LOW'))
        ):
            warnings.warn(
                f"You have chosen source as {self.source} and y_function as {self.y_function} Make sure this choice is intentional")

        # Create folder structure if it does not exists
        for folder in [self.data_folder, self.result_folder, self.models_folder, self.plot_folder]:
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)