from .content_based import get_priority_val, run_recommendation_pipeline
from typing import List, Dict, Any, Tuple

def preprocess_road_input(user_input: Dict[str, Any], 
                          binary_cols: List[str], 
                          continuous_cols: List[str]) -> Tuple[List[float], List[int]]:
    """
    Translates road running questionnaire responses into a standardized numerical 
    vector compatible with the Deep Autoencoder.
    
    This function applies domain-specific heuristic mapping to convert qualitative 
    user preferences (e.g., "Fast" pace, "Neutral" arch) into quantitative features. 
    It also implements feature masking to identify which features should be used 
    in similarity calculation based on what the user actually provided.
    
    Args:
        user_input (Dict[str, Any]): Raw questionnaire data from the frontend. 
            Expected keys include:
            - 'pace': str ('Easy', 'Steady', 'Fast')
            - 'arch_type': str ('Flat', 'Normal', 'High')
            - 'strike_pattern': str ('Heel', 'Mid', 'Forefoot')
            - 'foot_width': str ('Narrow', 'Regular', 'Wide')
            - 'season': str ('Summer', 'Spring & Fall', 'Winter')
            - 'orthotic_usage': str ('Yes', 'No')
            - 'running_purpose': str ('Daily', 'Tempo', 'Race')
            - 'cushion_preferences': str ('Soft', 'Balanced', 'Firm')
            - 'stability_need': str ('Neutral', 'Guided')
        binary_cols (List[str]): List of binary feature names from config.py 
            (e.g., ['lightweight', 'rocker', 'removable_insole', ...]). 
            These are encoded as 0.0 or 1.0.
        continuous_cols (List[str]): List of continuous feature names from 
            config.py (e.g., ['weight_lab_oz', 'drop_lab_mm', ...]). 
            These are normalized to [0, 1] range.
    
    Returns:
        Tuple[List[float], List[int]]: A tuple containing:
            - full_vector_raw (List[float]): The complete numerical vector 
              representing user preferences. Length matches total feature count 
              (typically 31 for road). Values are in [0, 1] range.
            - valid_indices (List[int]): Indices of features the user explicitly 
              provided. Used for masked cosine similarity to prevent noise from 
              default values. If empty (no user input), returns all indices.
    
    Feature Mapping Strategy:
        - Multi-source priority: Some features (e.g., strike pattern) consider 
          multiple inputs with priority ordering
        - Derived features: Some features are computed from others 
          (e.g., weight_lab_oz = 1.0 - lightweight)
        - Static features: Quality attributes (durability, breathability) default 
          to 1.0 (high quality) and are always included in similarity
        - Seasonal distribution: Season preference is distributed across three 
          binary features (summer/winter/all-season)
    
    Example:
        >>> user_input = {
        ...     'pace': 'Fast',
        ...     'arch_type': 'Normal',
        ...     'running_purpose': 'Race'
        ... }
        >>> binary_cols = ['lightweight', 'rocker', ...]
        >>> continuous_cols = ['weight_lab_oz', 'drop_lab_mm', ...]
        >>> vector, valid_idx = preprocess_road_input(user_input, binary_cols, continuous_cols)
        >>> len(vector)
        31
        >>> len(valid_idx)
        15  # Only features derived from pace/arch_type/running_purpose
        >>> vector[0]  # lightweight feature
        1.0  # Because pace='Fast'
    
    Notes:
        - All features default to neutral values (0.0 for binary, 0.5 for continuous) 
          if not derivable from user input
        - Feature masking (valid_indices) is critical for accuracy when user 
          provides sparse input
        - The mapping logic is empirically tuned based on running biomechanics 
          and shoe design principles
    """
    # Initialize all potential features to neutral/zero
    feats = {col: 0.0 for col in binary_cols + continuous_cols}
    
    # --- 1. HEURISTIC FEATURE MAPPING ---
    
    # Lightweight & Rocker logic: Mapping pace and purpose to shoe geometry
    feats['lightweight'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}})
    
    feats['rocker'] = get_priority_val(user_input, ['running_purpose'], 
        {'running_purpose': {'Race': 1.0, 'Tempo': 0.5, 'Daily': 0.0}})
    
    feats['removable_insole'] = get_priority_val(user_input, ['orthotic_usage'], 
        {'orthotic_usage': {'Yes': 1.0, 'No': 0.5}})
    
    # Purpose-based Pace distribution
    purp = user_input.get('running_purpose', 'Daily')
    feats['pace_daily_running'] = 1.0 if purp == 'Daily' else (0.5 if purp == 'Tempo' else 0.0)
    feats['pace_tempo'] = 1.0 if purp == 'Tempo' else 0.5
    feats['pace_competition'] = 1.0 if purp == 'Race' else (0.5 if purp == 'Tempo' else 0.0)

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

    # Midsole & Fit Preferences
    prio_soft = ['cushion_preferences', 'pace']
    feats['midsole_softness'] = get_priority_val(user_input, prio_soft, {
        'cushion_preferences': {'Soft': 1.0, 'Balanced': 0.6, 'Firm': 0.2}, 
        'pace': {'Easy': 1.0, 'Steady': 0.6, 'Fast': 0.2}})

    prio_width = ['stability_need', 'foot_width']
    feats['width_fit'] = get_priority_val(user_input, prio_width, {
        'stability_need': {'Neutral': 0.5, 'Guided': 0.2}, 
        'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1}})
    feats['toebox_width'] = get_priority_val(user_input, ['stability_need'], 
        {'stability_need': {'Neutral': 0.5, 'Guided': 0.2}})
    
    # Technical Specs: Stiffness and Plate Technology
    prio_stiff = ['arch_type', 'pace', 'running_purpose']
    feats['stiffness_scaled'] = get_priority_val(user_input, prio_stiff, {
        'arch_type': {'Flat': 0.0, 'Normal': 0.5, 'High': 0.5}, 
        'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}, 
        'running_purpose': {'Daily': 0.2, 'Tempo': 0.6, 'Race': 1}})

    feats['torsional_rigidity'] = get_priority_val(user_input, ['arch_type', 'pace'], {
        'arch_type': {'Flat': 1.0, 'Normal': 0.5, 'High': 0.5}, 
        'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})

    feats['heel_stiff'] = get_priority_val(user_input, ['arch_type'], 
        {'arch_type': {'Flat': 1.0, 'Normal': 0.6, 'High': 0.2}})

    # Configuration Sync: Ensure keys match config.py exactly
    prio_plate = ['pace', 'running_purpose']
    feats['plate_rock_plate'] = 0.5 
    feats['plate_carbon_plate'] = get_priority_val(user_input, prio_plate, {
        'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}, 
        'running_purpose': {'Daily': 0.5, 'Tempo': 0.5, 'Race': 1.0}})

    # Cushion Geometry (Stack Height)
    prio_stack = ['strike_pattern', 'pace', 'running_purpose']
    feats['heel_lab_mm'] = get_priority_val(user_input, prio_stack, {
        'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 
        'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['forefoot_lab_mm'] = get_priority_val(user_input, prio_stack, {
        'strike_pattern': {'Heel': 0.0, 'Mid': 0.5, 'Forefoot': 1.0}, 
        'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    # Derived Features
    feats['weight_lab_oz'] = 1.0 - feats['lightweight']
    
    # Static Quality Attributes (Default to high quality)
    feats['toebox_durability'] = 1.0
    feats['heel_durability'] = 1.0
    feats['outsole_durability'] = 1.0
    feats['breathability_scaled'] = 1.0

    # Environmental/Seasonal Mapping
    feats['season_summer'] = get_priority_val(user_input, ['season'], 
        {'season': {'Summer': 1.0, 'Spring & Fall': 0.5, 'Winter': 0.0}})
    feats['season_winter'] = get_priority_val(user_input, ['season'], 
        {'season': {'Summer': 0.0, 'Spring & Fall': 0.0, 'Winter': 1.0}})
    feats['season_all'] = get_priority_val(user_input, ['season'], 
        {'season': {'Summer': 0.5, 'Spring & Fall': 1.0, 'Winter': 0.0}})
    
    # --- 2. FEATURE MASKING LOGIC ---
    
    # Identify which inputs the user actually provided to focus the similarity search
    provided_inputs = {k for k, v in user_input.items() if v}
    
    # Mapping of shoe features to their questionnaire sources
    feature_sources = {
        'lightweight': ['pace'], 'rocker': ['running_purpose'], 'removable_insole': ['orthotic_usage'],
        'pace_daily_running': ['running_purpose'], 'pace_tempo': ['running_purpose'], 'pace_competition': ['running_purpose'],
        'arch_neutral': ['arch_type'], 'arch_stability': ['arch_type'], 'drop_lab_mm': ['pace'],
        'strike_heel': ['strike_pattern', 'pace'], 'strike_mid': ['strike_pattern', 'pace'], 'strike_forefoot': ['strike_pattern', 'pace'],
        'midsole_softness': ['cushion_preferences', 'pace'], 'width_fit': ['stability_need', 'foot_width'],
        'toebox_width': ['stability_need'], 'stiffness_scaled': ['arch_type', 'pace', 'running_purpose'],
        'torsional_rigidity': ['arch_type', 'pace'], 'heel_stiff': ['arch_type'],
        'plate_rock_plate': ['pace', 'running_purpose'], 'plate_carbon_plate': ['pace', 'running_purpose'],
        'heel_lab_mm': ['strike_pattern', 'pace'], 'forefoot_lab_mm': ['strike_pattern', 'pace'],
        'weight_lab_oz': ['pace'], 'season_summer': ['season'], 'season_winter': ['season'], 'season_all': ['season'],
        # Empty source lists indicate static features that are always valid in similarity calculation
        'toebox_durability': [], 'heel_durability': [], 'outsole_durability': [], 'breathability_scaled': []
    }
    
    all_cols = binary_cols + continuous_cols
    # Construct the raw vector following binary then continuous ordering
    full_vector_raw = [feats.get(c, 0.0) if c in binary_cols else feats.get(c, 0.5) for c in all_cols]
    
    # Determine valid indices: features with provided sources or static attributes
    valid_indices = []
    for i, col in enumerate(all_cols):
        sources = feature_sources.get(col, [])
        if not sources or any(src in provided_inputs for src in sources):
            valid_indices.append(i)
    
    return full_vector_raw, valid_indices or list(range(len(all_cols)))


def get_recommendations(user_input: Dict[str, Any], artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Executes the full road shoe recommendation pipeline from raw user input to 
    ranked recommendations.
    
    This is the main entry point for road shoe recommendations, orchestrating 
    the preprocessing and recommendation pipeline.
    
    Args:
        user_input (Dict[str, Any]): Raw questionnaire responses from the user. 
            See preprocess_road_input() for expected keys.
        artifacts (Dict[str, Any]): Pre-loaded model artifacts containing:
            - 'binary_cols' (List[str]): Binary feature names
            - 'continuous_cols' (List[str]): Continuous feature names
            - 'df_data' (pd.DataFrame): Shoe metadata with cluster labels
            - 'encoder_model' (tf.keras.Model): Trained Autoencoder
            - 'kmeans_model' (sklearn.cluster.KMeans): Trained K-Means
            - 'X_combined_data' (np.ndarray): Scaled feature matrix
    
    Returns:
        List[Dict[str, Any]]: Top 10 recommended road shoes with metadata:
            [
                {
                    'shoe_id': 'R045',
                    'name': 'Nike Vaporfly 3',
                    'brand': 'Nike',
                    'cluster': 2,
                    'match_score': 0.97,
                    ... (other shoe attributes)
                },
                ...
            ]
    
    Workflow:
        1. Preprocess user input into numerical vector
        2. Identify valid feature indices for masking
        3. Pass to recommendation pipeline (encoding → clustering → similarity)
        4. Return ranked results
    
    Example:
        >>> artifacts = load_cb_artifacts("model_artifacts/road")
        >>> user_input = {
        ...     'pace': 'Fast',
        ...     'arch_type': 'Normal',
        ...     'running_purpose': 'Race',
        ...     'strike_pattern': 'Forefoot'
        ... }
        >>> recommendations = get_recommendations(user_input, artifacts)
        >>> recommendations[0]['name']
        'Nike Vaporfly 3'
        >>> recommendations[0]['match_score']
        0.97
    """
    full_vector, valid_idx = preprocess_road_input(
        user_input, 
        artifacts['binary_cols'], 
        artifacts['continuous_cols']
    )
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)