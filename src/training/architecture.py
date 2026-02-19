"""
Sonix-ML Neural Architecture
----------------------------
Defines the Deep Autoencoder structure used for dimensionality reduction 
and feature extraction of shoe attributes.

The architecture follows a symmetrical bottleneck design:
Input (N-dim) -> Compressed (32D) -> Compressed (16D) -> Latent (8D) -> Reconstruction.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
from typing import Tuple, List

def build_autoencoder(input_dim: int, 
                      encoding_dims: List[int] = [32, 16, 8], 
                      dropout_rate: float = 0.3) -> Tuple[Model, Model]:
    """
    Constructs a Deep Autoencoder and a standalone Encoder model.
    
    The Encoder is used during inference to project user preferences into 
    the Latent Space for cluster routing.

    Args:
        input_dim: Number of input features (varies between Road and Trail).
        encoding_dims: List of neurons for each progressive compression layer.
        dropout_rate: Regularization rate to prevent overfitting.

    Returns:
        Tuple: (Full Autoencoder Model, Inference-only Encoder Model).
    """
    
    # --- ENCODER SECTION ---
    # Responsibility: Compress raw features into a dense latent representation.
    input_layer = layers.Input(shape=(input_dim,), name="feature_input")
    x = input_layer
    
    for i, dim in enumerate(encoding_dims):
        x = layers.Dense(dim, activation='relu', name=f"encoder_dense_{i}")(x)
        # BatchNormalization stabilizes training and speeds up convergence
        x = layers.BatchNormalization(name=f"encoder_bn_{i}")(x)
        # Dropout ensures the model doesn't memorize specific training examples
        x = layers.Dropout(dropout_rate, name=f"encoder_dropout_{i}")(x)

    latent_space = x  # The bottleneck (8D representation)

    # --- DECODER SECTION ---
    # Responsibility: Reconstruct the original input from the latent space.
    # This phase is only used during training (MSE loss calculation).
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        x = layers.Dense(dim, activation='relu', name=f"decoder_dense_{i}")(x)
        x = layers.BatchNormalization(name=f"decoder_bn_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"decoder_dropout_{i}")(x)

    # Output uses sigmoid activation to match normalized input range (0 to 1)
    output_layer = layers.Dense(input_dim, activation='sigmoid', name="reconstruction_output")(x)

    # --- MODEL COMPILATION ---
    
    # 1. Full Autoencoder (Trainable)
    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="Sonix_Autoencoder")
    
    # 2. Standalone Encoder (Inference-only)
    # This model shares weights with the autoencoder but stops at the latent space.
    encoder = Model(inputs=input_layer, outputs=latent_space, name="Sonix_Encoder")

    # Adam optimizer with Mean Squared Error (MSE) loss is the standard for reconstruction tasks.
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae'] # Mean Absolute Error for human-readable performance tracking
    )
    
    return autoencoder, encoder