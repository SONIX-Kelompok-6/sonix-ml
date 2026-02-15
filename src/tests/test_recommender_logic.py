import pytest
from src.recommender.content_based import get_priority_val

def test_get_priority_val_mapping():
    """Verify that user inputs map to the correct heuristic weights."""
    user_input = {'terrain': 'Rocky'}
    mapping = {'terrain': {'Light': 0.0, 'Mixed': 0.5, 'Rocky': 1.0}}
    
    # Test high priority match
    val = get_priority_val(user_input, ['terrain'], mapping)
    assert val == 1.0

def test_get_priority_val_fallback():
    """Ensure the function returns a neutral default (0.5) for missing inputs."""
    user_input = {'terrain': None}
    mapping = {'terrain': {'Light': 0.0, 'Mixed': 0.5, 'Rocky': 1.0}}
    
    val = get_priority_val(user_input, ['terrain'], mapping)
    assert val == 0.5

def test_get_priority_val_multi_source():
    """Test priority weight calculation from multiple input sources."""
    user_input = {'pace': 'Fast', 'terrain': 'Muddy'}
    mapping = {
        'pace': {'Fast': 1.0, 'Slow': 0.0},
        'terrain': {'Muddy': 1.0, 'Dry': 0.0}
    }
    
    # Combined logic: (1.0 + 1.0) / 2 = 1.0
    val = get_priority_val(user_input, ['pace', 'terrain'], mapping)
    assert val == 1.0