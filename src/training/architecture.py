import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

def build_autoencoder(input_dim: int, encoding_dims=[32, 16, 8], dropout_rate=0.3):
    """
    Membangun model Autoencoder dan Encoder terpisah.
    Input_dim akan menyesuaikan apakah itu Road atau Trail.
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    x = input_layer
    for dim in encoding_dims:
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    latent = x  # Latent space (8D)

    # Decoder
    for dim in reversed(encoding_dims[:-1]):
        x = layers.Dense(dim, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

    output_layer = layers.Dense(input_dim, activation='sigmoid')(x)

    # Full Model
    autoencoder = Model(input_layer, output_layer)
    # Model untuk ambil vector latent (Inference)
    encoder = Model(input_layer, latent)

    autoencoder.compile(
        optimizer=optimizers.Adam(0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return autoencoder, encoder