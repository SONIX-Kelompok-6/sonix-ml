from .content_based import get_priority_val, run_recommendation_pipeline

def preprocess_trail_input(user_input, binary_cols, continuous_cols):
    """Rules khusus untuk TRAIL."""
    feats = {col: 0.0 for col in binary_cols + continuous_cols}
    
    feats['terrain_light']     = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 1.0, 'Mixed': 0.5, 'Rocky': 0.0, 'Muddy': 0.0}})
    feats['terrain_moderate']  = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 0.5}})
    feats['terrain_technical'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.0, 'Mixed': 0.5, 'Rocky': 1.0, 'Muddy': 1.0}})

    feats['shock_absorption']  = get_priority_val(user_input, ['rock_sensitive', 'terrain'], 
                                                  {'rock_sensitive': {'Yes': 1.0, 'No': 0.0}, 
                                                   'terrain': {'Light': 0.2, 'Mixed': 0.6, 'Rocky': 1.0, 'Muddy': 0.0}})

    feats['energy_return'] = 1.0

    feats['traction_scaled'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})

    feats['arch_neutral']   = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 0.0, 'Normal': 0.8, 'High': 1.0}})
    feats['arch_stability'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 1.0, 'Normal': 0.2, 'High': 0.0}})

    feats['drop_lab_mm'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})

    prio_strike = ['strike_pattern', 'pace']
    feats['strike_heel']     = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['strike_mid']      = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 0.5, 'Mid': 1.0, 'Forefoot': 0.5}, 'pace': {'Easy': 0.5, 'Steady': 1.0, 'Fast': 0.5}})
    feats['strike_forefoot'] = get_priority_val(user_input, prio_strike, {'strike_pattern': {'Heel': 0.0, 'Mid': 0.0, 'Forefoot': 1.0}, 'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    feats['midsole_softness'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 1.0, 'Steady': 0.6, 'Fast': 0.2}})

    feats['toebox_durability']  = 1.0
    feats['heel_durability']    = 1.0
    feats['outsole_durability'] = 1.0
    feats['breathability']      = 1.0

    feats['plate_rock_plate'] = get_priority_val(user_input, ['pace', 'terrain'], 
                                                 {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 0.5}, 
                                                  'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 1.0, 'Muddy': 1.0}})

    feats['plate_carbon_plate'] = get_priority_val(user_input, ['pace', 'terrain'], 
                                                   {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}, 
                                                    'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 0.5}})

    feats['width_fit']    = get_priority_val(user_input, ['foot_width'], {'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1.0}})
    feats['toebox_width'] = get_priority_val(user_input, ['foot_width'], {'foot_width': {'Narrow': 0.2, 'Regular': 0.6, 'Wide': 1.0}})

    feats['stiffness_scaled'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})

    feats['torsional_rigidity'] = get_priority_val(user_input, ['arch_type', 'pace'], 
                                                   {'arch_type': {'Flat': 1.0, 'Normal': 0.5, 'High': 0.5}, 
                                                    'pace': {'Easy': 0.2, 'Steady': 0.6, 'Fast': 1.0}})

    feats['heel_stiff'] = get_priority_val(user_input, ['arch_type'], {'arch_type': {'Flat': 1.0, 'Normal': 0.6, 'High': 0.2}})

    feats['lug_depth'] = get_priority_val(user_input, ['terrain'], {'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})

    prio_stack_h = ['strike_pattern', 'pace', 'terrain']
    feats['heel_lab_mm'] = get_priority_val(user_input, prio_stack_h, 
                                            {'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 
                                             'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}, 
                                             'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 1.0, 'Muddy': 1.0}})

    prio_stack_f = ['strike_pattern', 'pace', 'terrain']
    feats['forefoot_lab_mm'] = get_priority_val(user_input, prio_stack_f, 
                                                {'strike_pattern': {'Heel': 0.0, 'Mid': 0.5, 'Forefoot': 1.0}, 
                                                 'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}, 
                                                 'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 0.5}})

    feats['season_summer'] = get_priority_val(user_input, ['season'], {'season': {'Summer': 1.0, 'Spring, Fall': 0.5, 'Winter': 0.0}})
    feats['season_winter'] = get_priority_val(user_input, ['season'], {'season': {'Summer': 0.0, 'Spring, Fall': 0.0, 'Winter': 1.0}})
    feats['season_all']    = get_priority_val(user_input, ['season'], {'season': {'Summer': 0.5, 'Spring, Fall': 1.0, 'Winter': 0.0}})

    feats['removable_insole'] = get_priority_val(user_input, ['orthotic_usage'], {'orthotic_usage': {'Yes': 1.0, 'No': 0.5}})

    feats['waterproof'] = get_priority_val(user_input, ['water_resistance', 'terrain'], 
                                           {'water_resistance': {'Waterproof': 1.0, 'Water Repellent': 0.5}, 
                                            'terrain': {'Light': 0.5, 'Mixed': 0.5, 'Rocky': 0.5, 'Muddy': 1.0}})

    feats['water_repellent'] = get_priority_val(user_input, ['water_resistance', 'terrain'], 
                                                {'water_resistance': {'Waterproof': 1.0, 'Water Repellent': 1.0}, 
                                                 'terrain': {'Light': 0.5, 'Mixed': 1.0, 'Rocky': 0.5, 'Muddy': 1.0}})

    feats['lightweight'] = get_priority_val(user_input, ['pace'], {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}})


    provided_inputs = {k for k, v in user_input.items() if v}  # Track which inputs user provided

    feature_sources = {
        'terrain_light': ['terrain'], 
        'terrain_moderate': ['terrain'], 
        'terrain_technical': ['terrain'],
        'shock_absorption': ['rock_sensitive', 'terrain'],
        'traction_scaled': ['terrain'],
        'arch_neutral': ['arch_type'], 
        'arch_stability': ['arch_type'],
        'drop_lab_mm': ['pace'],
        'strike_heel': ['strike_pattern', 'pace'], 
        'strike_mid': ['strike_pattern', 'pace'], 
        'strike_forefoot': ['strike_pattern', 'pace'],
        'midsole_softness': ['pace'],
        'plate_rock_plate': ['pace', 'terrain'],
        'plate_carbon_plate': ['pace', 'terrain'],
        'width_fit': ['foot_width'], 
        'toebox_width': ['foot_width'],
        'stiffness_scaled': ['pace'],
        'torsional_rigidity': ['arch_type', 'pace'],
        'heel_stiff': ['arch_type'],
        'lug_depth': ['terrain'],
        'heel_lab_mm': ['strike_pattern', 'pace', 'terrain'],
        'forefoot_lab_mm': ['strike_pattern', 'pace', 'terrain'],
        'season_summer': ['season'], 
        'season_winter': ['season'], 
        'season_all': ['season'],
        'removable_insole': ['orthotic_usage'],
        'waterproof': ['water_resistance', 'terrain'],
        'water_repellent': ['water_resistance', 'terrain'],
        'lightweight': ['pace'],
        'energy_return': [], 'toebox_durability': [], 'heel_durability': [], 'outsole_durability': [], 'breathability': []
    }

    all_cols = binary_cols + continuous_cols
    full_vector_raw = []
    for col in binary_cols:
        full_vector_raw.append(feats.get(col, 0.0))
    for col in continuous_cols:
        full_vector_raw.append(feats.get(col, 0.5))

    valid_indices = []
    for i, col in enumerate(all_cols):
        sources = feature_sources.get(col, [])
        # Fitur dianggap valid jika SALAH SATU source input-nya ada
        if any(src in provided_inputs for src in sources):
            valid_indices.append(i)
            
    if not valid_indices:
        valid_indices = list(range(len(all_cols)))

    return full_vector_raw, valid_indices

def get_recommendations(user_input, artifacts):
    # 1. Panggil fungsi preprocess khusus Trail
    full_vector, valid_idx = preprocess_trail_input(
        user_input, 
        artifacts['binary_cols'], 
        artifacts['continuous_cols']
    )
    
    # 2. Lempar ke pipeline umum
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)