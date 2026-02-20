"""
Road Recommender Module
-----------------------
Translates user questionnaire responses into a numerical vector tailored for 
road running shoe features. Refactored to use unified dictionary lookups 
to minimize cyclomatic complexity while preserving 100% of the original logic.
"""

from .content_based import get_priority_val, run_recommendation_pipeline
from typing import List, Dict, Any, Tuple

def preprocess_road_input(user_input: Dict[str, Any], 
                          binary_cols: List[str], 
                          continuous_cols: List[str]) -> Tuple[List[float], List[int]]:
    """
    Translates road running preferences into a standardized numerical vector.
    
    Uses dictionary-based mapping to eliminate ternary operators and if-else 
    branching, ensuring the lowest possible cyclomatic complexity.

    Args:
        user_input (Dict[str, Any]): Raw questionnaire data from the frontend.
        binary_cols (List[str]): List of binary feature names.
        continuous_cols (List[str]): List of continuous feature names.

    Returns:
        Tuple[List[float], List[int]]: 
            - The full numerical vector (N-dimensional).
            - A list of indices for masked similarity calculation.
    """
    all_cols = binary_cols + continuous_cols
    feats = {col: 0.0 for col in all_cols}
    
    # 1. Heuristic Priority Mappings
    feats['lightweight'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}})
    feats['rocker'] = get_priority_val(user_input, ['running_purpose'], {'running_purpose': {'Race': 1.0, 'Tempo': 0.5, 'Daily': 0.0}})
    feats['removable_insole'] = get_priority_val(user_input, ['orthotic_usage'], {'orthotic_usage': {'Yes': 1.0, 'No': 0.5}})
    
    # 2. Unified Dictionary Lookups (Replacing If-Else Ternaries)
    purp = user_input.get('running_purpose', 'Daily')
    feats['pace_daily_running'] = {'Daily': 1.0, 'Tempo': 0.5}.get(purp, 0.0)
    feats['pace_tempo'] = {'Tempo': 1.0}.get(purp, 0.5)
    feats['pace_competition'] = {'Race': 1.0, 'Tempo': 0.5}.get(purp, 0.0)

    feats['arch_neutral'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 0.0, 'Normal': 0.8, 'High': 1.0}})
    feats['arch_stability'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 1.0, 'Normal': 0.2, 'High': 0.0}})
    feats['drop_lab_mm'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})

    prio_strike = ['strike_pattern', 'pace']
    feats['strike_heel'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['strike_mid'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 0.5, 'Mid': 1.0, 'Forefoot': 0.5}, 'pace': {'Easy': 0.5, 'Steady': 1.0, 'Fast': 0.5}})
    feats['strike_forefoot'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 0.0, 'Mid': 0.0, 'Forefoot': 1.0}, 'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    feats['midsole_softness'] = get_priority_val(user_input, ['cushion_preferences', 'pace'], {'cushion_preferences': {'Soft': 1.0, 'Balanced': 0.6, 'Firm': 0.2}, 'pace': {'Easy': 1.0, 'Steady': 0.6, 'Fast': 0.2}})

    feats['width_fit'] = get_priority_val(user_input, ['stability_need', 'foot_width'], {'stability_need': {'Neutral': 0.5, 'Guided': 0.2}, 'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1}})
    feats['toebox_width'] = get_priority_val(user_input, ['stability_need'], {'stability_need': {'Neutral': 0.5, 'Guided': 0.2}})
    
    feats['stiffness_scaled'] = get_priority_val(user_input, ['arch_type', 'pace', 'running_purpose'], {'arch_type': {'Flat': 0.0, 'Normal': 0.5, 'High': 0.5}, 'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}, 'running_purpose': {'Daily': 0.2, 'Tempo': 0.6, 'Race': 1}})
    feats['torsional_rigidity'] = get_priority_val(user_input, ['arch_type', 'pace'], {'arch_type': {'Flat': 1.0, 'Normal': 0.5, 'High': 0.5}, 'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})
    feats['heel_stiff'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 1.0, 'Normal': 0.6, 'High': 0.2}})

    feats['plate_rock_plate'] = 0.5 
    feats['plate_carbon_plate'] = get_priority_val(user_input, ['pace', 'running_purpose'], {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}, 'running_purpose': {'Daily': 0.5, 'Tempo': 0.5, 'Race': 1.0}})

    feats['heel_lab_mm'] = get_priority_val(user_input, ['strike_pattern', 'pace'], {'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['forefoot_lab_mm'] = get_priority_val(user_input, ['strike_pattern', 'pace'], {'strike_pattern': {'Heel': 0.0, 'Mid': 0.5, 'Forefoot': 1.0}, 'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    feats['weight_lab_oz'] = 1.0 - feats.get('lightweight', 0.5)
    
    feats['season_summer'] = get_priority_val(user_input, ['season'], {'season': {'Summer': 1.0, 'Spring & Fall': 0.5, 'Winter': 0.0}})
    feats['season_winter'] = get_priority_val(user_input, ['season'], {'season': {'Summer': 0.0, 'Spring & Fall': 0.0, 'Winter': 1.0}})
    feats['season_all'] = get_priority_val(user_input, ['season'], {'season': {'Summer': 0.5, 'Spring & Fall': 1.0, 'Winter': 0.0}})
    
    # Static Column Assignment
    for col in ['toebox_durability', 'heel_durability', 'outsole_durability', 'breathability_scaled']:
        feats[col] = 1.0

    # 3. Vector and Masking Setup
    binary_set = set(binary_cols)
    full_vector_raw = [feats.get(c, 0.0 if c in binary_set else 0.5) for c in all_cols]
    
    provided_inputs = {k for k, v in user_input.items() if v}
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
        'weight_lab_oz': ['pace'], 'season_summer': ['season'], 'season_winter': ['season'], 'season_all': ['season']
    }
    
    valid_indices = [
        i for i, col in enumerate(all_cols)
        if not feature_sources.get(col) or not set(feature_sources[col]).isdisjoint(provided_inputs)
    ]
    
    return full_vector_raw, valid_indices or list(range(len(all_cols)))

def get_recommendations(user_input: Dict[str, Any], artifacts: Dict[str, Any]) -> List[str]:
    """Wrapper entry point for road recommendation."""
    full_vector, valid_idx = preprocess_road_input(user_input, artifacts['binary_cols'], artifacts['continuous_cols'])
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)