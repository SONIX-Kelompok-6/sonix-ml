"""
Trail Recommender Module
------------------------
Translates user questionnaire responses into a numerical vector tailored for 
trail running shoe features. Implements data-driven feature mapping and 
masking logic to maintain low cyclomatic complexity.
"""

from .content_based import get_priority_val, run_recommendation_pipeline
from typing import List, Dict, Any, Tuple

def preprocess_trail_input(user_input: Dict[str, Any], 
                           binary_cols: List[str], 
                           continuous_cols: List[str]) -> Tuple[List[float], List[int]]:
    """
    Translates trail running questionnaire responses into a standardized numerical vector.
    
    Applies heuristic mapping to convert qualitative trail preferences (like terrain 
    and traction) into quantitative features. Utilizes dictionary lookups and set 
    operations to eliminate branching and maintain O(1) cyclomatic complexity.

    Args:
        user_input (Dict[str, Any]): Raw questionnaire data from the frontend.
        binary_cols (List[str]): List of binary feature names defined in configurations.
        continuous_cols (List[str]): List of continuous feature names.

    Returns:
        Tuple[List[float], List[int]]: 
            - The full numerical vector compatible with the Deep Autoencoder.
            - A list of indices representing active user preferences for masked similarity.
    """
    all_cols = binary_cols + continuous_cols
    feats = {col: 0.0 for col in all_cols}
    
    feats['terrain_light'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 1.0, 'Mixed': 0.5, 'Rocky': 0.0, 'Muddy': 0.0}})
    feats['terrain_moderate'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 0.5}})
    feats['terrain_technical'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.0, 'Mixed': 0.5, 'Rocky': 1.0, 'Muddy': 1.0}})

    feats['lug_dept_mm'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})
    feats['traction_scaled'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})

    feats['shock_absorption'] = get_priority_val(user_input, ['rock_sensitive', 'terrain'], {'rock_sensitive': {'Yes': 1.0, 'No': 0.0}, 'terrain': {'Light': 0.2, 'Mixed': 0.6, 'Rocky': 1.0, 'Muddy': 0.0}})
    feats['energy_return'] = 1.0 
    
    feats['arch_neutral'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 0.0, 'Normal': 0.8, 'High': 1.0}})
    feats['arch_stability'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 1.0, 'Normal': 0.2, 'High': 0.0}})

    feats['drop_lab_mm'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})

    prio_strike = ['strike_pattern', 'pace']
    feats['strike_heel'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['strike_mid'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 0.5, 'Mid': 1.0, 'Forefoot': 0.5}, 'pace': {'Easy': 0.5, 'Steady': 1.0, 'Fast': 0.5}})
    feats['strike_forefoot'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 0.0, 'Mid': 0.0, 'Forefoot': 1.0}, 'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    feats['midsole_softness'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 1.0, 'Steady': 0.6, 'Fast': 0.2}})

    feats['plate_rock_plate'] = get_priority_val(user_input, ['pace', 'terrain'], {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 0.5}, 'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 1.0, 'Muddy': 1.0}})
    feats['plate_carbon_plate'] = get_priority_val(user_input, ['pace', 'terrain'], {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}, 'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 0.5}})

    feats['width_fit'] = get_priority_val(user_input, ['foot_width'], {'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1.0}})
    feats['toebox_width'] = get_priority_val(user_input, ['foot_width'], {'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1.0}})

    feats['stiffness_scaled'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})
    feats['torsional_rigidity'] = get_priority_val(user_input, ['arch_type', 'pace'], {'arch_type': {'Flat': 1.0, 'Normal': 0.5, 'High': 0.5}, 'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})
    feats['heel_stiff'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 1.0, 'Normal': 0.6, 'High': 0.2}})

    prio_stack = ['strike_pattern', 'pace', 'terrain']
    feats['heel_lab_mm'] = get_priority_val(user_input, prio_stack, {'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}, 'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 1.0, 'Muddy': 1.0}})
    feats['forefoot_lab_mm'] = get_priority_val(user_input, prio_stack, {'strike_pattern': {'Heel': 0.0, 'Mid': 0.5, 'Forefoot': 1.0}, 'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}, 'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 0.5}})

    feats['waterproof'] = get_priority_val(user_input, ['water_resistance', 'terrain'], {'water_resistance': {'Waterproof': 1.0, 'Water Repellent': 0.5}, 'terrain': {'Muddy': 1.0}})
    feats['water_repellent'] = get_priority_val(user_input, ['water_resistance', 'terrain'], {'water_resistance': {'Waterproof': 1.0, 'Water Repellent': 1.0}, 'terrain': {'Mixed': 1.0, 'Muddy': 1.0}})

    feats.update({
        'toebox_durability': 1.0, 'heel_durability': 1.0, 
        'outsole_durability': 1.0, 'breathability_scaled': 1.0
    })
    feats['lightweight'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}})

    provided_inputs = {k for k, v in user_input.items() if v}
    
    # Feature sources converted to sets for O(1) intersection checks
    feature_sources = {
        'terrain_light': {'terrain'}, 'terrain_moderate': {'terrain'}, 'terrain_technical': {'terrain'},
        'shock_absorption': {'rock_sensitive', 'terrain'}, 'traction_scaled': {'terrain'},
        'arch_neutral': {'arch_type'}, 'arch_stability': {'arch_type'},
        'drop_lab_mm': {'pace'}, 'midsole_softness': {'pace'},
        'strike_heel': {'strike_pattern', 'pace'}, 'strike_mid': {'strike_pattern', 'pace'}, 'strike_forefoot': {'strike_pattern', 'pace'},
        'plate_rock_plate': {'pace', 'terrain'}, 'plate_carbon_plate': {'pace', 'terrain'},
        'width_fit': {'foot_width'}, 'toebox_width': {'foot_width'},
        'stiffness_scaled': {'pace'}, 'torsional_rigidity': {'arch_type', 'pace'},
        'heel_stiff': {'arch_type'}, 'lug_dept_mm': {'terrain'},
        'heel_lab_mm': {'strike_pattern', 'pace', 'terrain'}, 'forefoot_lab_mm': {'strike_pattern', 'pace', 'terrain'},
        'season_summer': {'season'}, 'season_winter': {'season'}, 'season_all': {'season'},
        'removable_insole': {'orthotic_usage'}, 'lightweight': {'pace'},
        'waterproof': {'water_resistance', 'terrain'}, 'water_repellent': {'water_resistance', 'terrain'}
    }

    binary_set = set(binary_cols)
    full_vector_raw = [feats.get(c, 0.0) if c in binary_set else feats.get(c, 0.5) for c in all_cols]
    
    valid_indices = [
        i for i, col in enumerate(all_cols)
        if not feature_sources.get(col) or not provided_inputs.isdisjoint(feature_sources[col])
    ]
    
    return full_vector_raw, valid_indices or list(range(len(all_cols)))

def get_recommendations(user_input: Dict[str, Any], artifacts: Dict[str, Any]) -> List[str]:
    """
    Wrapper function to execute the full trail recommendation pipeline.
    
    Args:
        user_input (Dict[str, Any]): Raw user preferences.
        artifacts (Dict[str, Any]): Dictionary of loaded models and metadata configurations.

    Returns:
        List[str]: A list containing the top recommended trail shoe IDs.
    """
    full_vector, valid_idx = preprocess_trail_input(
        user_input, 
        artifacts['binary_cols'], 
        artifacts['continuous_cols']
    )
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)