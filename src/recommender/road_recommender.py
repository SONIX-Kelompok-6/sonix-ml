from .content_based import get_priority_val, run_recommendation_pipeline
from typing import List, Dict, Any, Tuple

def preprocess_road_input(user_input: Dict[str, Any], 
                          binary_cols: List[str], 
                          continuous_cols: List[str]) -> Tuple[List[float], List[int]]:
    """
    Translates road running questionnaire responses into a standardized numerical vector.
    
    This function applies heuristic mapping to convert qualitative preferences into 
    quantitative features compatible with the Deep Autoencoder's input layer. 
    It also identifies 'valid' features based on user participation for the masking process.

    Args:
        user_input: Raw questionnaire data from the frontend.
        binary_cols: List of binary feature names defined in config.py.
        continuous_cols: List of continuous feature names defined in config.py.

    Returns:
        Tuple: (The full numerical vector, List of indices representing active user preferences).
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
    Wrapper function to execute the full road recommendation pipeline.
    """
    full_vector, valid_idx = preprocess_road_input(
        user_input, 
        artifacts['binary_cols'], 
        artifacts['artifacts_continuous_cols']
    )
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)