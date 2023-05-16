import keras
import keras.layers as layers
keras.metrics.RootMeanSquaredError()
keras.metrics.MeanAbsoluteError()
keras.metrics.MeanAbsolutePercentageError()
keras.metrics.MeanSquaredError()
keras.metrics.MeanSquaredLogarithmicError()
from pathlib import Path


from keras.applications import EfficientNetB0


# Another model to test could be EfficientNet model

class MLP:
    def conv1D_model(self, input_shape):
        input_layer = keras.layers.Input(input_shape)

        conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
        conv1 = keras.layers.BatchNormalization()(conv1)
        conv1 = keras.layers.ReLU()(conv1)

        conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
        conv2 = keras.layers.BatchNormalization()(conv2)
        conv2 = keras.layers.ReLU()(conv2)

        conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
        conv3 = keras.layers.BatchNormalization()(conv3)
        conv3 = keras.layers.ReLU()(conv3)

        gap = keras.layers.GlobalAveragePooling1D()(conv3)

        output_layer = keras.layers.Dense(1)(gap)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def lstm_model(self, input_shape):
        input_layer = keras.layers.Input(input_shape)

        lstm1 = keras.layers.LSTM(units=64, return_sequences=True)(input_layer)
        lstm2 = keras.layers.LSTM(units=64)(lstm1)

        output_layer = keras.layers.Dense(1)(lstm2)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def gru_model(self, input_shape):
        print(input_shape)
        input_layer = keras.layers.Input(input_shape)

        gru1 = keras.layers.GRU(units=64, return_sequences=True)(input_layer)
        gru2 = keras.layers.GRU(units=64)(gru1)

        output_layer = keras.layers.Dense(1)(gru2)

        return keras.models.Model(inputs=input_layer, outputs=output_layer)

    def build_model(self, x_train, model_type):
        if (model_type == 'gru_model' or model_type == 'lstm_model' or model_type == 'conv1D_model'):
            model = eval(f"self.{model_type}(input_shape=x_train.shape[1:])")
        else:
            raise ValueError(
                f"{model_type} is an incorrect model_type. The value should be: gru_model, lstm_model or conv1D_model.")
        keras.utils.plot_model(model, show_shapes=True)
        return model

    def train_model(self, config, model_name, model, x_train, y_train):
        epochs = config.transformer_setting['epoc']
        batch_size = config.transformer_setting['batch_size']
        # print(f"epochs: {epochs} - type: {type(epochs)}")
        # print(f"batch_size: {batch_size} - type: {type(batch_size)}")
        model_folder = config.models_folder
        print(f"model_name = {model_name}")
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_folder / Path(model_name + " best_model.h5"), save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
        ]
        model.compile(
            optimizer="adam",
            loss="mean_squared_error",
            metrics=[keras.metrics.RootMeanSquaredError(name='rmse')],
        )
        history = model.fit(
            x_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=0.2,
            verbose=1,
        )
        return history, model
