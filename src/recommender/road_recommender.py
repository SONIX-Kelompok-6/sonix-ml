from .content_based import get_priority_val, run_recommendation_pipeline

def preprocess_road_input(user_input, binary_cols, continuous_cols):
    """Aturan khusus untuk mengonversi kuesioner Road menjadi vector angka."""
    feats = {col: 0.0 for col in binary_cols + continuous_cols}
    
    # --- 1. PEMETAAN FITUR (MAPPING) ---
    feats['lightweight'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}})
    feats['rocker'] = get_priority_val(user_input, ['running_purpose'], 
        {'running_purpose': {'Race': 1.0, 'Tempo': 0.5, 'Daily': 0.0}})
    feats['removable_insole'] = get_priority_val(user_input, ['orthotic_usage'], 
        {'orthotic_usage': {'Yes': 1.0, 'No': 0.5}})
    
    purp = user_input.get('running_purpose', 'Daily')
    feats['pace_daily_running'] = 1.0 if purp == 'Daily' else (0.5 if purp == 'Tempo' else 0.0)
    feats['pace_tempo'] = 1.0 if purp == 'Tempo' else 0.5
    feats['pace_competition'] = 1.0 if purp == 'Race' else (0.5 if purp == 'Tempo' else 0.0)

    feats['arch_neutral'] = get_priority_val(user_input, ['arch_type'], 
        {'arch_type': {'Flat': 0.0, 'Normal': 0.8, 'High': 1.0}})
    feats['arch_stability'] = get_priority_val(user_input, ['arch_type'], 
        {'arch_type': {'Flat': 1.0, 'Normal': 0.2, 'High': 0.0}})
    
    feats['drop_lab_mm'] = get_priority_val(user_input, ['pace'], 
        {'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})

    # Strike Pattern
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

    # Cushion & Width
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
    
    # Stiffness & Rigidity
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

    # FIX: Penyesuaian nama kolom plate agar sesuai config.py
    prio_plate = ['pace', 'running_purpose']
    feats['plate_rock_plate'] = 0.5 
    feats['plate_carbon_plate'] = get_priority_val(user_input, prio_plate, {
        'pace': {'Easy': 0.5, 'Steady': 0.5, 'Fast': 1.0}, 
        'running_purpose': {'Daily': 0.5, 'Tempo': 0.5, 'Race': 1.0}})

    # Stack Height
    prio_stack = ['strike_pattern', 'pace', 'running_purpose']
    feats['heel_lab_mm'] = get_priority_val(user_input, prio_stack, {
        'strike_pattern': {'Heel': 1.0, 'Mid': 0.5, 'Forefoot': 0.0}, 
        'pace': {'Easy': 1.0, 'Steady': 0.5, 'Fast': 0.0}})
    feats['forefoot_lab_mm'] = get_priority_val(user_input, prio_stack, {
        'strike_pattern': {'Heel': 0.0, 'Mid': 0.5, 'Forefoot': 1.0}, 
        'pace': {'Easy': 0.0, 'Steady': 0.5, 'Fast': 1.0}})

    feats['weight_lab_oz'] = 1.0 - feats['lightweight']
    
    # FIX: Penyesuaian nama kolom durability & breathability
    feats['toebox_durability'] = 1.0
    feats['heel_durability'] = 1.0
    feats['outsole_durability'] = 1.0
    feats['breathability_scaled'] = 1.0

    # Season
    feats['season_summer'] = get_priority_val(user_input, ['season'], 
        {'season': {'Summer': 1.0, 'Spring & Fall': 0.5, 'Winter': 0.0}})
    feats['season_winter'] = get_priority_val(user_input, ['season'], 
        {'season': {'Summer': 0.0, 'Spring & Fall': 0.0, 'Winter': 1.0}})
    feats['season_all'] = get_priority_val(user_input, ['season'], 
        {'season': {'Summer': 0.5, 'Spring & Fall': 1.0, 'Winter': 0.0}})
    
    # --- 2. LOGIKA MASKING (LENGKAP) ---
    provided_inputs = {k for k, v in user_input.items() if v}
    
    # FIX: Daftar lengkap pemetaan fitur ke kuesioner sumber
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
        'toebox_durability': [], 'heel_durability': [], 'outsole_durability': [], 'breathability_scaled': []
    }
    
    all_cols = binary_cols + continuous_cols
    full_vector_raw = [feats.get(c, 0.0) if c in binary_cols else feats.get(c, 0.5) for c in all_cols]
    
    # Masking: Kolom valid jika input sumbernya diisi user ATAU kolom tersebut statis (tanpa sumber)
    valid_indices = []
    for i, col in enumerate(all_cols):
        sources = feature_sources.get(col, [])
        if not sources or any(src in provided_inputs for src in sources):
            valid_indices.append(i)
    
    return full_vector_raw, valid_indices or list(range(len(all_cols)))

def get_recommendations(user_input, artifacts):
    full_vector, valid_idx = preprocess_road_input(user_input, artifacts['binary_cols'], artifacts['continuous_cols'])
    return run_recommendation_pipeline(full_vector, valid_idx, artifacts)