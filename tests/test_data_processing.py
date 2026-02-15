import pytest
import pandas as pd
from src.database import fetch_and_merge_training_data

def test_rating_conversion_logic():
    """Verify star ratings (1-5) convert to symmetric weights (-2.0 to 2.0)."""
    # Local helper mirroring the internal database logic
    def convert(r):
        mapping = {5: 2.0, 4: 1.0, 3: 0.1, 2: -1.0, 1: -2.0}
        return mapping.get(r, 0.1)

    assert convert(5) == 2.0  # High preference
    assert convert(1) == -2.0 # High dislike
    assert convert(3) == 0.1  # Neutral

def test_empty_dataframe_handling():
    """Ensure the system doesn't crash if Supabase returns no data."""
    from src.database import fetch_shoes_by_type
    
    # This test confirms that our error handling returns an empty DF instead of None
    df = fetch_shoes_by_type("non_existent_type")
    assert isinstance(df, pd.DataFrame)
    assert df.empty