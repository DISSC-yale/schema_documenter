import json
import os
import tempfile
from pathlib import Path
import time

import pandas as pd
import pytest
import pyreadstat

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


def test_get_schema_stata(sample_stata_file):
    """Test schema extraction from Stata file with metadata."""
    documenter = SchemaDocumenter(sample_stata_file)
    schema_info = documenter.get_schema(Path(sample_stata_file))

    # Basic file info checks
    assert schema_info["file_name"] == Path(sample_stata_file).name
    assert schema_info["file_type"] == ".dta"
    assert schema_info["column_count"] == 4
    assert not schema_info.get("missing_columns", False)
    assert not schema_info.get("error")

    # Check column information
    assert len(schema_info["columns"]) == 4
    for col in schema_info["columns"]:
        assert "name" in col
        assert "dtype" in col
        assert "non_null_count" in col
        assert "null_count" in col
        assert "unique_values" in col
        assert "sample_values" in col

    # Check specific column names
    column_names = [col["name"] for col in schema_info["columns"]]
    expected_names = ['numeric_var', 'string_var', 'date_var', 'categorical_var']
    assert sorted(column_names) == sorted(expected_names)


def test_stata_encoding_handling(temp_dir):
    """Test handling of different Stata file encodings."""
    # Create test data with special characters
    data = {
        'test_var': ['Test with special chars: äöüß']
    }
    df = pd.DataFrame(data)
    
    # Create Stata file
    file_path = temp_dir / "test.dta"
    pyreadstat.write_dta(df, file_path)
    
    # Try to read it back with different encodings
    documenter = SchemaDocumenter(str(file_path))
    schema_info = documenter.get_schema(file_path)
    
    # Verify successful reading
    assert not schema_info.get("error")
    assert len(schema_info["columns"]) == 1
    assert schema_info["columns"][0]["name"] == "test_var"
    
    # Verify the special characters are preserved
    sample_values = schema_info["columns"][0]["sample_values"]
    assert "äöüß" in sample_values[0]


def test_is_schema_needed(temp_dir):
    """Test schema file update checking."""
    documenter = SchemaDocumenter(temp_dir)
    
    # Create a test data file
    data_file = temp_dir / "test.csv"
    df = pd.DataFrame({'test': [1, 2, 3]})
    df.to_csv(data_file)
    
    # Create a schema file
    schema_file = data_file.with_suffix(data_file.suffix + ".schema")
    schema_info = documenter.get_schema(data_file)
    documenter.save_schema(schema_info, schema_file)
    
    # Test cases
    assert not documenter.is_schema_needed(data_file, schema_file)  # Same timestamp
    
    # Modify data file
    time.sleep(1)  # Ensure different timestamp
    df.to_csv(data_file)
    assert documenter.is_schema_needed(data_file, schema_file)  # Data file newer
    
    # Non-existent schema file
    non_existent_schema = temp_dir / "nonexistent.schema"
    assert documenter.is_schema_needed(data_file, non_existent_schema)


def test_get_schema_empty_file(temp_dir):
    """Test handling of empty files."""
    # Create an empty file
    file_path = temp_dir / "empty.parquet"
    file_path.touch()
    
    documenter = SchemaDocumenter(str(file_path))
    schema_info = documenter.get_schema(file_path)
    
    assert schema_info["error"] == "File is empty"
    assert schema_info["missing_columns"] is True
    assert schema_info["column_count"] == 0


def test_get_schema_invalid_json(temp_dir):
    """Test handling of invalid JSON files."""
    # Create an invalid JSON file
    file_path = temp_dir / "invalid.json"
    with open(file_path, "w") as f:
        f.write("{ invalid json")
    
    documenter = SchemaDocumenter(str(file_path))
    schema_info = documenter.get_schema(file_path)
    
    assert "error" in schema_info
    assert schema_info["missing_columns"] is True
    assert schema_info["column_count"] == 0


def test_get_schema_nonexistent_file():
    """Test handling of non-existent files."""
    documenter = SchemaDocumenter("nonexistent.csv")
    schema_info = documenter.get_schema(Path("nonexistent.csv"))
    
    assert "error" in schema_info
    assert schema_info["missing_columns"] is True
    assert schema_info["column_count"] == 0


def test_get_schema_unsupported_file(temp_dir):
    """Test handling of unsupported file types."""
    # Create a text file
    file_path = temp_dir / "test.txt"
    with open(file_path, "w") as f:
        f.write("Some text")
    
    documenter = SchemaDocumenter(str(file_path))
    assert not documenter.is_supported_file(file_path)


def test_save_schema_invalid_path(temp_dir):
    """Test handling of invalid paths when saving schema."""
    documenter = SchemaDocumenter(temp_dir)
    schema_info = {
        "file_name": "test.csv",
        "file_type": ".csv",
        "columns": []
    }
    
    # Try to save to a non-existent directory
    invalid_path = temp_dir / "nonexistent_dir" / "schema.json"
    documenter.save_schema(schema_info, invalid_path)
    
    # The parent directory should be created
    assert invalid_path.parent.exists()
    assert invalid_path.exists()


def test_process_directory_empty(temp_dir):
    """Test processing an empty directory."""
    documenter = SchemaDocumenter(temp_dir)
    documenter.process_directory()
    # Should not raise any errors


def test_process_directory_mixed_files(temp_dir):
    """Test processing a directory with mixed file types."""
    # Create various files with actual data
    df = pd.DataFrame({'test': [1, 2, 3]})
    df.to_csv(temp_dir / "test.csv", index=False)
    (temp_dir / "test.txt").touch()  # Unsupported
    (temp_dir / "test.json").write_text('{"test": [1, 2, 3]}')
    
    documenter = SchemaDocumenter(temp_dir)
    documenter.process_directory()
    
    # Check schema files were created only for supported types
    assert (temp_dir / "test.csv.schema").exists()
    assert (temp_dir / "test.json.schema").exists()
    assert not (temp_dir / "test.txt.schema").exists()


def test_schema_documenter_with_subdirectories(temp_dir):
    """Test processing nested directories."""
    # Create a nested directory structure
    subdir = temp_dir / "subdir"
    subdir.mkdir()
    
    # Create files in both root and subdir with actual data
    df = pd.DataFrame({'test': [1, 2, 3]})
    df.to_csv(temp_dir / "root.csv", index=False)
    df.to_csv(subdir / "sub.csv", index=False)
    
    documenter = SchemaDocumenter(temp_dir)
    documenter.process_directory()
    
    # Check schema files were created in both directories
    assert (temp_dir / "root.csv.schema").exists()
    assert (subdir / "sub.csv.schema").exists()


def test_get_schema_with_null_values(temp_dir):
    """Test handling of null values in data."""
    # Create a CSV with null values
    data = pd.DataFrame({
        'col1': [1, None, 3],
        'col2': [None, None, None],
        'col3': ['a', 'b', None]
    })
    file_path = temp_dir / "null_test.csv"
    data.to_csv(file_path, index=False)
    
    documenter = SchemaDocumenter(str(file_path))
    schema_info = documenter.get_schema(file_path)
    
    # Check null counts
    col_info = {col['name']: col for col in schema_info['columns']}
    assert col_info['col1']['null_count'] == 1
    assert col_info['col2']['null_count'] == 3
    assert col_info['col3']['null_count'] == 1
