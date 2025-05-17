tf.random.set_seed(0)
# YOUR CODE HERE
import keras_tuner as kt

def build_model_tuner(hp):
    """Build model function for hyperparameter tuning."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(0)
    
    model = tf.keras.Sequential()

    # Hyperparameter tuning for learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-2, step=1e-6)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    
    # Add the Dense layer
    model.add(tf.keras.layers.Dense(
        units=1,        # output dim
        input_shape=[num_features],  # input dim
        use_bias=True,               # use a bias (intercept) param
        kernel_initializer=tf.ones_initializer,  # initialize params to 1      
        bias_initializer=tf.keras.initializers.Ones(),    # initialize bias to 1
    ))

    model.compile(optimizer=optimizer, loss='mse')

    return model

# Initialize the tuner
tuner = kt.Hyperband(
    build_model_tuner,
    objective='val_loss',
    max_epochs=20,
    factor=3,
    hyperband_iterations=2,
    directory='tuner',
    project_name='automobile_price_model_tuning'
)

# Search for the best hyperparameters
tuner.search(X_train_std, Y_train_std, epochs=20, validation_data=(X_val_std, Y_val_std))