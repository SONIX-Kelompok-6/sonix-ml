from .content_based import get_priority_val, run_recommendation_pipeline
from typing import List, Dict, Any, Tuple

def preprocess_trail_input(user_input: Dict[str, Any], 
                           binary_cols: List[str], 
                           continuous_cols: List[str]) -> Tuple[List[float], List[int]]:
    """
    Translates trail running questionnaire responses into a standardized numerical 
    vector compatible with the Deep Autoencoder.
    
    This function handles trail-specific heuristic mapping for features like terrain 
    difficulty, traction (lug depth), protection (shock absorption, rock plate), 
    and weather resistance. It also implements feature masking to identify which 
    features should be used in similarity calculation based on user input.
    
    Args:
        user_input (Dict[str, Any]): Raw questionnaire data from the frontend. 
            Expected keys include:
            - 'pace': str ('Easy', 'Steady', 'Fast')
            - 'arch_type': str ('Flat', 'Normal', 'High')
            - 'strike_pattern': str ('Heel', 'Mid', 'Forefoot')
            - 'foot_width': str ('Narrow', 'Regular', 'Wide')
            - 'season': str ('Summer', 'Spring & Fall', 'Winter')
            - 'orthotic_usage': str ('Yes', 'No')
            - 'terrain': str ('Light', 'Mixed', 'Rocky', 'Muddy')
            - 'rock_sensitive': str ('Yes', 'No')
            - 'water_resistance': str ('Waterproof', 'Water Repellent')
        binary_cols (List[str]): List of binary feature names from config.py 
            (e.g., ['lightweight', 'waterproof', 'water_repellent', ...]). 
            These are encoded as 0.0 or 1.0.
        continuous_cols (List[str]): List of continuous feature names from 
            config.py (e.g., ['weight_lab_oz', 'lug_dept_mm', ...]). 
            These are normalized to [0, 1] range.
    
    Returns:
        Tuple[List[float], List[int]]: A tuple containing:
            - full_vector_raw (List[float]): The complete numerical vector 
              representing user preferences. Length matches total feature count 
              (typically 36 for trail). Values are in [0, 1] range.
            - valid_indices (List[int]): Indices of features the user explicitly 
              provided. Used for masked cosine similarity to prevent noise from 
              default values. If empty (no user input), returns all indices.
    
    Feature Mapping Strategy:
        - Terrain distribution: Terrain difficulty is distributed across three 
          features (light/moderate/technical) to capture intensity spectrum
        - Traction scaling: Lug depth correlates with terrain muddiness and 
          technical difficulty
        - Protection features: Shock absorption and rock plates prioritized for 
          rocky/technical terrain
        - Weather resistance: Waterproof/water-repellent features activated by 
          water_resistance preference and muddy terrain
        - Static features: Quality attributes (durability, energy return) default 
          to 1.0 and are always included in similarity
    
    Example:
        >>> user_input = {
        ...     'pace': 'Steady',
        ...     'terrain': 'Rocky',
        ...     'rock_sensitive': 'Yes',
        ...     'water_resistance': 'Water Repellent'
        ... }
        >>> binary_cols = ['lightweight', 'waterproof', ...]
        >>> continuous_cols = ['lug_dept_mm', 'traction_scaled', ...]
        >>> vector, valid_idx = preprocess_trail_input(user_input, binary_cols, continuous_cols)
        >>> len(vector)
        36
        >>> len(valid_idx)
        18  # Only features derived from pace/terrain/rock_sensitive/water_resistance
        >>> vector[4]  # shock_absorption feature
        1.0  # Because rock_sensitive='Yes' and terrain='Rocky'
    
    Notes:
        - All features default to neutral values (0.0 for binary, 0.5 for continuous) 
          if not derivable from user input
        - Feature masking (valid_indices) is critical for accuracy when user 
          provides sparse input
        - The mapping logic is empirically tuned based on trail running biomechanics 
          and off-road shoe design principles
        - Trail shoes emphasize protection and traction over speed optimization
    """
    # Initialize feature space to neutral defaults
    feats = {col: 0.0 for col in binary_cols + continuous_cols}
    
    # --- 1. TERRAIN & TRACTION MAPPING ---
    
    # Distribution of terrain intensity across three specialized features
    feats['terrain_light'] = get_priority_val(user_input, ['terrain'], 
        {'terrain': {'Light': 1.0, 'Mixed': 0.5, 'Rocky': 0.0, 'Muddy': 0.0}})
    feats['terrain_moderate'] = get_priority_val(user_input, ['terrain'], 
        {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 0.5}})
    feats['terrain_technical'] = get_priority_val(user_input, ['terrain'], 
        {'terrain': {'Light': 0.0, 'Mixed': 0.5, 'Rocky': 1.0, 'Muddy': 1.0}})

    # Lug depth mapping: Muddy and Mixed terrain require deeper traction
    feats['lug_dept_mm'] = get_priority_val(user_input, ['terrain'], 
        {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})

    feats['traction_scaled'] = get_priority_val(user_input, ['terrain'], 
        {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})

    # --- 2. PROTECTION & GEOMETRY ---

    # Shock absorption: High priority if user is rock-sensitive or running technical trails
    feats['shock_absorption'] = get_priority_val(user_input, ['rock_sensitive', 'terrain'], 
        {'rock_sensitive': {'Yes': 1.0, 'No': 0.0}, 
         'terrain': {'Light': 0.2, 'Mixed': 0.6, 'Rocky': 1.0, 'Muddy': 0.0}})

    feats['energy_return'] = 1.0 # Standard trail high-performance default
    
    # Foot Morphology (Arch Support)
    feats['arch_neutral'] = get_priority_val(user_input, ['arch_type'], 
        {'arch_type': {'Flat': 0.0, 'Normal': 0.8, 'High': 1.0}})
    feats['arch_stability'] = get_priority_val(user_input, ['arch_type'], 
        {'arch_type': {'Flat': 1.0, 'Normal': 0.2, 'High': 0.0}})

    feats['drop_lab_mm'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})

    # Biomechanical Strike Patterns
    prio_strike = ['strike_pattern', 'pace']
    feats['strike_heel'] = get_priority_val(user_input, prio_strike, {
        'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 
        'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['strike_mid'] = get_priority_val(user_input, prio_strike, {
        'strike_pattern': {'Heel': 0.5, 'Mid': 1.0, 'Forefoot': 0.5}, 
        'pace': {'Easy': 0.5, 'Steady': 1.0, 'Fast': 0.5}})
    feats['strike_forefoot'] = get_priority_val(user_input, prio_strike, {
        'strike_pattern': {'Heel': 0.0, 'Mid': 0.0, 'Forefoot': 1.0}, 
        'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    feats['midsole_softness'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 1.0, 'Steady': 0.6, 'Fast': 0.2}})

    # --- 3. PLATES & STABILITY ---

    feats['plate_rock_plate'] = get_priority_val(user_input, ['pace', 'terrain'], 
        {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 0.5}, 
         'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 1.0, 'Muddy': 1.0}})
    
    feats['plate_carbon_plate'] = get_priority_val(user_input, ['pace', 'terrain'], 
        {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}, 
         'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 0.5}})

    # Fit and Rigidity attributes
    feats['width_fit'] = get_priority_val(user_input, ['foot_width'], 
        {'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1.0}})
    feats['toebox_width'] = get_priority_val(user_input, ['foot_width'], 
        {'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1.0}})

    feats['stiffness_scaled'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})

    feats['torsional_rigidity'] = get_priority_val(user_input, ['arch_type', 'pace'], 
        {'arch_type': {'Flat': 1.0, 'Normal': 0.5, 'High': 0.5}, 
         'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})

    feats['heel_stiff'] = get_priority_val(user_input, ['arch_type'], 
        {'arch_type': {'Flat': 1.0, 'Normal': 0.6, 'High': 0.2}})

    # --- 4. CUSHION & PROTECTION ---

    prio_stack = ['strike_pattern', 'pace', 'terrain']
    feats['heel_lab_mm'] = get_priority_val(user_input, prio_stack, {
        'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 
        'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}, 
        'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 1.0, 'Muddy': 1.0}})
    
    feats['forefoot_lab_mm'] = get_priority_val(user_input, prio_stack, {
        'strike_pattern': {'Heel': 0.0, 'Mid': 0.5, 'Forefoot': 1.0}, 
        'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}, 
        'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 0.5}})

    # Weather Resistance
    feats['waterproof'] = get_priority_val(user_input, ['water_resistance', 'terrain'], 
        {'water_resistance': {'Waterproof': 1.0, 'Water Repellent': 0.5}, 
         'terrain': {'Muddy': 1.0}})
    
    feats['water_repellent'] = get_priority_val(user_input, ['water_resistance', 'terrain'], 
        {'water_resistance': {'Waterproof': 1.0, 'Water Repellent': 1.0}, 
         'terrain': {'Mixed': 1.0, 'Muddy': 1.0}})

    # Static Quality Attributes
    feats['toebox_durability'] = 1.0
    feats['heel_durability'] = 1.0
    feats['outsole_durability'] = 1.0
    feats['breathability_scaled'] = 1.0
    feats['lightweight'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}})

    # --- 5. FEATURE MASKING LOGIC ---

    provided_inputs = {k for k, v in user_input.items() if v}
    
    # Mapping trail features to their specific questionnaire sources for similarity focus
    feature_sources = {
        'terrain_light': ['terrain'], 'terrain_moderate': ['terrain'], 'terrain_technical': ['terrain'],
        'shock_absorption': ['rock_sensitive', 'terrain'], 'traction_scaled': ['terrain'],
        'arch_neutral': ['arch_type'], 'arch_stability': ['arch_type'],
        'drop_lab_mm': ['pace'], 'midsole_softness': ['pace'],
        'strike_heel': ['strike_pattern', 'pace'], 'strike_mid': ['strike_pattern', 'pace'], 'strike_forefoot': ['strike_pattern', 'pace'],
        'plate_rock_plate': ['pace', 'terrain'], 'plate_carbon_plate': ['pace', 'terrain'],
        'width_fit': ['foot_width'], 'toebox_width': ['foot_width'],
        'stiffness_scaled': ['pace'], 'torsional_rigidity': ['arch_type', 'pace'],
        'heel_stiff': ['arch_type'], 'lug_dept_mm': ['terrain'],
        'heel_lab_mm': ['strike_pattern', 'pace', 'terrain'], 'forefoot_lab_mm': ['strike_pattern', 'pace', 'terrain'],
        'season_summer': ['season'], 'season_winter': ['season'], 'season_all': ['season'],
        'removable_insole': ['orthotic_usage'], 'lightweight': ['pace'],
        'waterproof': ['water_resistance', 'terrain'], 'water_repellent': ['water_resistance', 'terrain'],
        # Static features remain valid across all searches to maintain quality baseline
        'energy_return': [], 'toebox_durability': [], 'heel_durability': [], 'outsole_durability': [], 'breathability_scaled': []
    }

    all_cols = binary_cols + continuous_cols
    full_vector_raw = [feats.get(c, 0.0) if c in binary_cols else feats.get(c, 0.5) for c in all_cols]
    
    # Calculate valid indices for the Masked Cosine Similarity
    valid_indices = []
    for i, col in enumerate(all_cols):
        sources = feature_sources.get(col, [])
        if not sources or any(src in provided_inputs for src in sources):
            valid_indices.append(i)
    
    return full_vector_raw, valid_indices or list(range(len(all_cols)))


def get_recommendations(user_input: Dict[str, Any], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes the full trail shoe recommendation pipeline from raw user input to 
    ranked recommendations.
    
    This is the main entry point for trail shoe recommendations, orchestrating 
    the preprocessing and recommendation pipeline with trail-specific features.
    
    Args:
        user_input (Dict[str, Any]): Raw questionnaire responses from the user. 
            See preprocess_trail_input() for expected keys. Trail-specific keys 
            include 'terrain', 'rock_sensitive', and 'water_resistance'.
        artifacts (Dict[str, Any]): Pre-loaded model artifacts containing:
            - 'binary_cols' (List[str]): Binary feature names for trail shoes
            - 'continuous_cols' (List[str]): Continuous feature names for trail shoes
            - 'df_data' (pd.DataFrame): Trail shoe metadata with cluster labels
            - 'encoder_model' (tf.keras.Model): Trained Autoencoder for trail
            - 'kmeans_model' (sklearn.cluster.KMeans): Trained K-Means for trail
            - 'X_combined_data' (np.ndarray): Scaled feature matrix for trail shoes
    
    Returns:
        List[Dict[str, Any]]: Top 10 recommended trail shoes with metadata:
            [
                {
                    'shoe_id': 'T012',
                    'name': 'Hoka Speedgoat 5',
                    'brand': 'Hoka',
                    'cluster': 4,
                    'match_score': 0.94,
                    'lug_dept_mm': 5.0,
                    'waterproof': True,
                    ... (other shoe attributes)
                },
                ...
            ]
    
    Workflow:
        1. Preprocess user input into numerical vector (trail-specific features)
        2. Identify valid feature indices for masking
        3. Pass to recommendation pipeline (encoding → clustering → similarity)
        4. Return ranked results
    
    Example:
        >>> artifacts = load_cb_artifacts("model_artifacts/trail")
        >>> user_input = {
        ...     'pace': 'Steady',
        ...     'terrain': 'Rocky',
        ...     'rock_sensitive': 'Yes',
        ...     'water_resistance': 'Water Repellent',
        ...     'arch_type': 'Normal'
        ... }
        >>> recommendations = get_recommendations(user_input, artifacts)
        >>> recommendations[0]['name']
        'Hoka Speedgoat 5'
        >>> recommendations[0]['match_score']
        0.94
        >>> recommendations[0]['lug_dept_mm']
        5.0
    
    Notes:
        - Trail recommendations prioritize protection, traction, and durability 
          over pure speed optimization
        - The model accounts for terrain difficulty, weather conditions, and 
          user sensitivity to rock/root impacts
    """
    full_vector, valid_idx = preprocess_trail_input(
        user_input, 
        artifacts['binary_cols'], 
        artifacts['continuous_cols']
    )
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)