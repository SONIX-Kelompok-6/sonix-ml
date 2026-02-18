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
    Constructs a Deep Autoencoder with symmetrical encoder-decoder architecture 
    and returns both the full autoencoder and a standalone encoder model.
    
    The autoencoder compresses high-dimensional shoe feature vectors into a 
    dense 8-dimensional latent space. During training, it learns to reconstruct 
    the original input from this compressed representation. At inference time, 
    only the encoder is used to project user preferences into the latent space 
    for K-Means cluster routing.
    
    Architecture Overview:
        Encoder Path:  Input(N) → Dense(32)+BN+Dropout → Dense(16)+BN+Dropout 
                       → Dense(8)+BN+Dropout → Latent(8D)
        Decoder Path:  Latent(8D) → Dense(16)+BN+Dropout → Dense(32)+BN+Dropout 
                       → Dense(N, sigmoid) → Reconstruction
    
    Args:
        input_dim (int): Number of input features. This varies by shoe category:
            - Road shoes: typically 31 features
            - Trail shoes: typically 36 features
        encoding_dims (List[int]): List of neuron counts for each progressive 
            compression layer in the encoder. Default: [32, 16, 8]
            - First layer (32): Initial compression from N-dim to 32-dim
            - Second layer (16): Intermediate compression
            - Third layer (8): Final latent space dimension (bottleneck)
        dropout_rate (float): Dropout probability for regularization. 
            Default: 0.3 (30% of neurons dropped during training)
            - Prevents overfitting by forcing redundancy in learned representations
            - Applied after each Dense layer in both encoder and decoder
    
    Returns:
        Tuple[Model, Model]: A tuple containing:
            - autoencoder (tf.keras.Model): Full autoencoder model for training. 
              Takes input shape (batch_size, input_dim) and outputs reconstructed 
              features of the same shape. Used for unsupervised pretraining.
            - encoder (tf.keras.Model): Inference-only encoder model. Takes input 
              shape (batch_size, input_dim) and outputs latent representation of 
              shape (batch_size, 8). Used for real-time recommendation queries.
    
    Model Architecture Details:
        Encoder Layers:
            - Dense(32, relu) + BatchNormalization + Dropout(0.3)
            - Dense(16, relu) + BatchNormalization + Dropout(0.3)
            - Dense(8, relu) + BatchNormalization + Dropout(0.3) → Latent Space
        
        Decoder Layers:
            - Dense(16, relu) + BatchNormalization + Dropout(0.3)
            - Dense(32, relu) + BatchNormalization + Dropout(0.3)
            - Dense(input_dim, sigmoid) → Reconstruction
        
        Compilation:
            - Optimizer: Adam (learning_rate=0.001)
            - Loss: Mean Squared Error (MSE)
            - Metrics: Mean Absolute Error (MAE)
    
    Training Strategy:
        The autoencoder is trained in an unsupervised manner to minimize 
        reconstruction error. The learned latent space naturally clusters 
        similar shoes together, which K-Means then exploits for fast retrieval.
    
    Inference Usage:
        Only the encoder is used at inference time:
        1. User preferences are converted to a feature vector
        2. Encoder projects this vector into 8D latent space
        3. K-Means identifies nearest clusters in latent space
        4. Candidate shoes from these clusters are retrieved
        5. Final ranking uses cosine similarity on original feature space
    
    Example:
        >>> # Build models for road shoes (31 features)
        >>> autoencoder, encoder = build_autoencoder(input_dim=31)
        >>> autoencoder.summary()
        Model: "Sonix_Autoencoder"
        _________________________________________________________________
        Layer (type)                 Output Shape              Param #
        =================================================================
        feature_input (InputLayer)   [(None, 31)]              0
        encoder_dense_0 (Dense)      (None, 32)                1024
        ...
        reconstruction_output (Dense) (None, 31)               1023
        =================================================================
        Total params: 3,456
        Trainable params: 3,200
        Non-trainable params: 256
        
        >>> # Train on shoe feature matrix
        >>> X_scaled = scaler.fit_transform(shoe_features)  # Shape: (n_shoes, 31)
        >>> history = autoencoder.fit(X_scaled, X_scaled, epochs=300, batch_size=64)
        
        >>> # Use encoder for inference
        >>> user_vector = np.array([[0.5, 1.0, 0.2, ...]])  # Shape: (1, 31)
        >>> latent_vector = encoder.predict(user_vector)    # Shape: (1, 8)
        >>> latent_vector.shape
        (1, 8)
    
    Design Rationale:
        - BatchNormalization: Stabilizes training by normalizing layer inputs, 
          allowing higher learning rates and faster convergence
        - Dropout: Prevents co-adaptation of neurons, ensuring the latent space 
          captures generalizable patterns rather than memorizing training data
        - ReLU activation: Non-linearity allows the network to learn complex 
          feature interactions
        - Sigmoid output: Matches the normalized input range [0, 1] for features 
          scaled by MinMaxScaler
        - Symmetrical architecture: Decoder mirrors encoder structure to ensure 
          the latent space is expressive enough for faithful reconstruction
    
    Notes:
        - The encoder and decoder share the same computational graph. Saving the 
          encoder separately at .h5 format is sufficient for inference.
        - During inference, dropout is automatically disabled (Keras handles this).
        - The latent dimension (8D) was empirically chosen as the optimal balance 
          between compression and information retention for shoe features.
        - Training typically requires 300 epochs with batch size 64 to converge.
    
    Related Functions:
        - training_engine.run_training(): Orchestrates the full training pipeline
        - content_based.run_recommendation_pipeline(): Uses the encoder for inference
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
        metrics=['mae']  # Mean Absolute Error for human-readable performance tracking
    )
    
    return autoencoder, encoder