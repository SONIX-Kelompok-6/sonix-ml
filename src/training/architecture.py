"""
Sonix-ML Neural Architecture Module
-----------------------------------
Defines the Deep Autoencoder structure utilized for dimensionality reduction 
and latent feature extraction of running shoe attributes.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from typing import Tuple, List

def build_autoencoder(input_dim: int, 
                      encoding_dims: List[int] = [32, 16, 8], 
                      dropout_rate: float = 0.3) -> Tuple[Model, Model]:
    """
    Constructs a symmetrical Deep Autoencoder and a standalone Encoder model.

    Args:
        input_dim (int): The number of dynamic input features.
        encoding_dims (List[int], optional): Neuron counts for progressive compression.
        dropout_rate (float, optional): Regularization fraction.

    Returns:
        Tuple[Model, Model]: The full Autoencoder and the standalone Encoder.
    """
    # --- ENCODER ---
    input_layer = layers.Input(shape=(input_dim,), name="feature_input")
    x = input_layer
    
    for i, dim in enumerate(encoding_dims):
        x = layers.Dense(dim, activation='relu', name=f"encoder_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"encoder_dropout_{i}")(x)

    latent_space = x 

    # --- DECODER ---
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        x = layers.Dense(dim, activation='relu', name=f"decoder_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"decoder_dropout_{i}")(x)

    output_layer = layers.Dense(input_dim, activation='sigmoid', name="reconstruction_output")(x)

    # --- COMPILATION ---
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="Sonix_Autoencoder")
    encoder = Model(inputs=input_layer, outputs=latent_space, name="Sonix_Encoder")

    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder, encoder