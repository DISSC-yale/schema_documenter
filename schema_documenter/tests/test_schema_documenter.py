import json
import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from schema_documenter import SchemaDocumenter


@pytest.fixture
def sample_csv_data():
    """Create a temporary CSV file with sample data."""
    data = {
        "name": ["Alice", "Bob", "Charlie"],
        "age": [25, 30, 35],
        "score": [85.5, 92.0, 78.5],
    }
    df = pd.DataFrame(data)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        df.to_csv(tmp.name, index=False)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_json_data():
    """Create a temporary JSON file with sample data."""
    data = [
        {"name": "Alice", "age": 25, "score": 85.5},
        {"name": "Bob", "age": 30, "score": 92.0},
        {"name": "Charlie", "age": 35, "score": 78.5},
    ]

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as tmp:
        json.dump(data, tmp)
        tmp_path = tmp.name

    yield tmp_path

    # Clean up
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


def test_schema_documenter_initialization():
    """Test SchemaDocumenter initialization."""
    documenter = SchemaDocumenter("test_path")
    assert documenter.input_path == Path("test_path")
    assert isinstance(documenter.supported_extensions, set)


def test_is_supported_file():
    """Test file extension support checking."""
    documenter = SchemaDocumenter("test_path")
    assert documenter.is_supported_file(Path("test.csv"))
    assert documenter.is_supported_file(Path("test.xlsx"))
    assert documenter.is_supported_file(Path("test.json"))
    assert not documenter.is_supported_file(Path("test.txt"))


def test_get_schema_csv(sample_csv_data):
    """Test schema extraction from CSV file."""
    documenter = SchemaDocumenter(sample_csv_data)
    schema_info = documenter.get_schema(Path(sample_csv_data))

    assert schema_info["file_name"] == Path(sample_csv_data).name
    assert schema_info["file_type"] == ".csv"
    assert schema_info["column_count"] == 3
    assert len(schema_info["columns"]) == 3

    # Check column information
    for col in schema_info["columns"]:
        assert "name" in col
        assert "dtype" in col
        assert "non_null_count" in col
        assert "null_count" in col
        assert "unique_values" in col
        assert "sample_values" in col


def test_get_schema_json(sample_json_data):
    """Test schema extraction from JSON file."""
    documenter = SchemaDocumenter(sample_json_data)
    schema_info = documenter.get_schema(Path(sample_json_data))

    assert schema_info["file_name"] == Path(sample_json_data).name
    assert schema_info["file_type"] == ".json"
    assert schema_info["column_count"] == 3
    assert len(schema_info["columns"]) == 3


def test_save_schema(sample_csv_data):
    """Test schema saving functionality."""
    documenter = SchemaDocumenter(sample_csv_data)
    schema_info = documenter.get_schema(Path(sample_csv_data))

    # Create a temporary directory for the test
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.schema"

        # Save the schema
        documenter.save_schema(schema_info, output_path)

        # Verify the saved schema file
        with open(output_path, "r") as f:
            saved_schema = json.load(f)
            assert saved_schema == schema_info


def test_process_directory(sample_csv_data):
    """Test directory processing functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy sample file to temporary directory
        import shutil

        shutil.copy(sample_csv_data, tmpdir)

        documenter = SchemaDocumenter(tmpdir)
        documenter.process_directory()

        # Verify schema file was created
        schema_file = Path(tmpdir) / (Path(sample_csv_data).name + ".schema")
        assert schema_file.exists()

        # Verify schema file content
        with open(schema_file, "r") as f:
            saved_schema = json.load(f)
            assert "columns" in saved_schema
            assert len(saved_schema["columns"]) == 3
