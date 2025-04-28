import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dataframe():
    """Create a sample pandas DataFrame."""
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "score": [85.5, 92.0, 78.5],
        }
    )


@pytest.fixture
def sample_csv_file(sample_dataframe, temp_dir):
    """Create a temporary CSV file with sample data."""
    file_path = temp_dir / "test.csv"
    sample_dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_json_file(sample_dataframe, temp_dir):
    """Create a temporary JSON file with sample data."""
    file_path = temp_dir / "test.json"
    sample_dataframe.to_json(file_path, orient="records")
    return file_path
