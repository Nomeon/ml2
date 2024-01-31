import tensorflow as tf


def create_cnn(spatial_input_shape, non_spatial_input_shape, num_actions, hidden_units):
    # Define input layer for spatial data
    spatial_input = tf.keras.layers.Input(shape=spatial_input_shape, name='spatial_input')

    # Build convolutional layers for feature extraction from spatial data
    x = spatial_input
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

    # Flatten before fully connected layers
    x = tf.keras.layers.Flatten()(x)

    # Define input layer for non-spatial data
    non_spatial_input = tf.keras.layers.Input(shape=non_spatial_input_shape, name='non_spatial_input')

    # Concatenate output from convolutional layers with non-spatial data
    merged_input = tf.keras.layers.concatenate([x, non_spatial_input], axis=-1)

    # Add fully connected layers for further processing
    x = tf.keras.layers.Dense(hidden_units, activation='relu', kernel_initializer='he_normal')(merged_input)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(hidden_units, activation='relu', kernel_initializer='he_normal')(x)

    # Output layer for action probabilities
    output_layer_actions = tf.keras.layers.Dense(num_actions, activation='softmax', name='output_layer_actions')(x)

    # Output layer for estimating baseline
    output_layer_baseline = tf.keras.layers.Dense(1, name='output_layer_baseline')(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=[spatial_input, non_spatial_input], outputs=[output_layer_actions, output_layer_baseline])
    model.compile(optimizer='adam', loss={'output_layer_actions': 'categorical_crossentropy', 'output_layer_baseline': 'mean_squared_error'}, metrics=['accuracy'])


    return model


# import numpy as np

# # Example usage:
# # Specify the input shape, number of channels, and other parameters
# num_channels = 10
# spatial_input_shape = (15, 15, num_channels) # width * height * num_channels
# non_spatial_input_shape = 5 # number of features not related to spatial data (health, current blast radius etc.)
# num_actions = 6 
# hidden_units = 64

# # Create the model
# model = create_cnn(spatial_input_shape, non_spatial_input_shape, num_actions, hidden_units)

# # Get input
# # Note that np.expand_dims is used since the first dimension normally represents the batch size, so spatial data input shape = (batch size, width, height, num_channels), non-spatial input shape = (batch size, number of non-spatial features)
# spatial_data = np.expand_dims(np.random.randint(0, 2, spatial_input_shape), axis=0) # Provide spatial data as input
# non_spatial_data = np.expand_dims(np.random.randint(0, 6, non_spatial_input_shape), axis=0) # Provide non-spatial data as input

# # Get action probabilities and estimated baseline
# action_probabilities, estimated_baseline = model([spatial_data, non_spatial_data])

# # Sample action
# sampled_action = np.random.choice(np.arange(num_actions), p=action_probabilities[0].numpy())

# # Get the estimated baseline
# baseline_estimate = estimated_baseline[0][0]