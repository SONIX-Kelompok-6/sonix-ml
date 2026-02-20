"""
Sonix-ML Neural Architecture Module
-----------------------------------
Defines the Deep Autoencoder structure utilized for dimensionality reduction 
and latent feature extraction of running shoe attributes.

The architecture strictly follows a symmetrical bottleneck design:
Input (N-dim) -> Compressed (32D) -> Compressed (16D) -> Latent (8D) -> Reconstruction.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from typing import Tuple, List

def build_autoencoder(input_dim: int, 
                      encoding_dims: List[int] = [32, 16, 8], 
                      dropout_rate: float = 0.3) -> Tuple[Model, Model]:
    """
    Constructs a symmetrical Deep Autoencoder and a standalone Encoder model.
    
    The standalone Encoder is designed for inference use, projecting high-dimensional 
    user preferences into a compressed Latent Space for efficient K-Means routing. 
    The full Autoencoder is utilized exclusively during the training phase.

    Args:
        input_dim (int): The number of input features (dynamically varies between Road and Trail).
        encoding_dims (List[int], optional): Sequence of neuron counts for progressive compression. 
                                             Defaults to [32, 16, 8].
        dropout_rate (float, optional): Regularization fraction to drop units and prevent overfitting. 
                                        Defaults to 0.3.

    Returns:
        Tuple[Model, Model]: 
            - The full Autoencoder Model compiled with Adam optimizer and MSE loss.
            - The standalone Encoder Model (sharing weights with the Autoencoder).
    """
    
    # --- ENCODER SECTION ---
    # Responsibility: Compress raw features into a dense latent representation.
    input_layer = layers.Input(shape=(input_dim,), name="feature_input")
    x = input_layer
    
    for i, dim in enumerate(encoding_dims):
        x = layers.Dense(dim, activation='relu', name=f"encoder_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"encoder_bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"encoder_dropout_{i}")(x)

    latent_space = x 

    # --- DECODER SECTION ---
    # Responsibility: Reconstruct original input. Used only during MSE calculation.
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        x = layers.Dense(dim, activation='relu', name=f"decoder_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"decoder_dropout_{i}")(x)

    # Output bounded between 0 and 1 matching MinMaxScaler range
    output_layer = layers.Dense(input_dim, activation='sigmoid', name="reconstruction_output")(x)

    # --- MODEL COMPILATION ---
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="Sonix_Autoencoder")
    encoder = Model(inputs=input_layer, outputs=latent_space, name="Sonix_Encoder")

    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder, encoder