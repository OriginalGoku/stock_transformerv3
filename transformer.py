import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import Config
from pathlib import Path
class Transformer:
    # Todo: Fix the transformer to use Config file
    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Attention and Normalization
        x = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(inputs, inputs)
        x = layers.Dropout(dropout)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        res = x + inputs

        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
        x = layers.Dropout(dropout)(x)
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        return x + res

    def build_model(self,
                    input_shape,
                    head_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout=0,
                    mlp_dropout=0,
                    pooling_type: str = 'GlobalAveragePooling1D'
                    ):
        inputs = keras.Input(shape=input_shape)
        x = inputs
        for _ in range(num_transformer_blocks):
            x = self.transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

        # todo: move all this to transformer_setting in config file
        if pooling_type == 'GlobalAveragePooling1D':
            x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
        elif pooling_type == 'GlobalMaxPooling1D':
            x = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
        elif pooling_type == 'AdaptviePooling':
            avg_pool = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
            max_pool = layers.GlobalMaxPooling1D(data_format="channels_first")(x)
            x = layers.Concatenate()([avg_pool, max_pool])
        elif pooling_type == 'LSTM':
            x = layers.Bidirectional(layers.LSTM(units=64))(x)
        elif pooling_type == 'Attention':
            attention = layers.Dense(1, activation='tanh')(x)
            attention = layers.Flatten()(attention)
            attention = layers.Activation('softmax')(attention)
            attention = layers.RepeatVector(x.shape[-1])(attention)
            attention = layers.Permute([2, 1])(attention)
            x = layers.Multiply()([x, attention])
            x = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(x)

        for dim in mlp_units:
            x = layers.Dense(dim, activation="relu")(x)
            x = layers.Dropout(mlp_dropout)(x)
        outputs = layers.Dense(1, activation=None)(x)
        return keras.Model(inputs, outputs)

    def train_model(self, config: Config, pooling_type: str, X_train, y_train):
        input_shape = X_train.shape[1:]

        model = self.build_model(
            input_shape,
            head_size=config.transformer_setting['head_size'],
            num_heads=config.transformer_setting['num_heads'],
            ff_dim=config.transformer_setting['ff_dim'],
            num_transformer_blocks=config.transformer_setting['num_transformer_blocks'],
            mlp_units=[config.transformer_setting['mlp_units']],
            mlp_dropout=config.transformer_setting['mlp_dropout'],
            dropout=config.transformer_setting['dropout'],
            pooling_type=pooling_type
        )

        # Todo: Fix Optimizer
        optimizer_name = config.transformer_setting['optimizer_choice'].default
        optimizer_class = getattr(keras.optimizers, optimizer_name)
        optimizer = optimizer_class(learning_rate=config.transformer_setting['learning_rate'])

        # print(global_config.transformer_setting['optimizer_choice'].default)

        # optimizer = keras.optimizers.RMSprop(learning_rate=config.transformer_setting['learning_rate'])

        # optimizer = eval(f"keras.optimizers.{config.transformer_setting['optimizer_choice']}(learning_rate=learning_rate)")
        # if optimizer_choice == "adam":
        #     optimizer = keras.optimizers.Adam(amsgrad=True, learning_rate=learning_rate)
        # elif optimizer_choice == "sgd":
        #     optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        # elif optimizer_choice == "rmsprop":
        #     optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)
        # elif optimizer_choice == "adagrad":
        #     optimizer = keras.optimizers.Adagrad(learning_rate=learning_rate)
        # elif optimizer_choice == "adadelta":
        #     optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)
        # elif optimizer_choice == "adamax":
        #     optimizer = keras.optimizers.Adamax(learning_rate=learning_rate)
        # elif optimizer_choice == "nadam":
        #     optimizer = keras.optimizers.Nadam(learning_rate=learning_rate)
        # elif optimizer_choice == "ftrl":
        #     optimizer = keras.optimizers.Ftrl(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss=config.transformer_setting['loss'],
                      metrics=config.transformer_setting['metrics'])

        # if print_summary:
        model.summary()

        # Todo: Fix this so it uses config file
        file_name = f"Transformer {pooling_type}.h5"
        model_name = config.models_folder / file_name
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                model_name, save_best_only=True, monitor="val_loss"
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.2, patience=2, min_lr=config.transformer_setting['min_learning_rate']
            ),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1),
        ]

        history = model.fit(
            X_train,
            y_train,
            validation_split=config.transformer_setting['validation_split'],
            epochs=config.transformer_setting['epoc'],
            batch_size=config.transformer_setting['batch_size'],
            callbacks=callbacks,
        )
        return history, model

    def evaluate_model(self, model, X_test, y_test):
        # Evaluate the model on the test dataset
        test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)

        print("Test Loss:", test_loss)
        print("Test Mean Absolute Error:", test_mae)

        return test_mae

    def load_model(self, model_name: Path):
        model = keras.models.load_model(model_name)
        return model