#!/usr/bin/env python3
"""
Dataset Schema Documenter

This module provides functionality to analyze datasets and create schema documentation files.
It supports multiple file formats and provides detailed information about each column/field.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # For Parquet files
import pyreadstat  # For Stata files
import rpy2.robjects as robjects  # For RDS files
from rpy2.robjects import pandas2ri

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class SchemaDocumenter:
    def __init__(self, input_path: str):
        """
        Initialize the SchemaDocumenter with the input path.

        Args:
            input_path (str): Path to the dataset file or directory containing datasets
        """
        self.input_path = Path(input_path)
        self.supported_extensions = {
            ".csv",
            ".xlsx",
            ".xls",
            ".json",
            ".dta",
            ".parquet",
            ".rds",
        }

    def is_supported_file(self, file_path: Path) -> bool:
        """
        Check if the file extension is supported.

        Args:
            file_path (Path): Path to the file

        Returns:
            bool: True if the file extension is supported, False otherwise
        """
        return file_path.suffix.lower() in self.supported_extensions

    def get_schema(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract schema information from a dataset file.

        Args:
            file_path (Path): Path to the dataset file

        Returns:
            Dict[str, Any]: Dictionary containing schema information
        """
        schema_info = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "file_type": file_path.suffix.lower(),
            "columns": [],
            "column_count": 0,
            "missing_columns": False,
        }

        try:
            if file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in {".xlsx", ".xls"}:
                df = pd.read_excel(file_path)
            elif file_path.suffix.lower() == ".json":
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    else:
                        df = pd.json_normalize(data)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON file {file_path}: {str(e)}")
                    schema_info["error"] = f"Invalid JSON format: {str(e)}"
                    return schema_info
                except Exception as e:
                    logger.error(f"Error processing JSON file {file_path}: {str(e)}")
                    schema_info["error"] = str(e)
                    return schema_info
            elif file_path.suffix.lower() == ".dta":
                try:
                    # Try reading with default encoding first
                    df, meta = pyreadstat.read_dta(file_path)
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with latin-1 encoding
                    logger.warning(
                        f"UTF-8 decoding failed for {file_path}, trying latin-1 encoding"
                    )
                    df, meta = pyreadstat.read_dta(file_path, encoding="latin-1")
                except Exception as e:
                    logger.error(f"Error reading Stata file {file_path}: {str(e)}")
                    schema_info["error"] = str(e)
                    return schema_info
            elif file_path.suffix.lower() == ".parquet":
                # Check if file is empty
                if os.path.getsize(file_path) == 0:
                    logger.warning(f"Parquet file {file_path} is empty")
                    schema_info["error"] = "File is empty"
                    return schema_info
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".rds":
                # Convert RDS to pandas DataFrame
                pandas2ri.activate()
                r_data = robjects.r["readRDS"](str(file_path))
                # Use the correct conversion method for rpy2
                df = pandas2ri.rpy2py(r_data)

            # Update column count
            schema_info["column_count"] = len(df.columns)

            # Extract column information
            for column in df.columns:
                try:
                    # Convert numpy types to Python native types for JSON serialization
                    sample_values = df[column].head(3).tolist()
                    sample_values = [
                        int(val) if isinstance(val, (np.integer, np.int64)) else val
                        for val in sample_values
                    ]

                    col_info = {
                        "name": column,
                        "dtype": str(df[column].dtype),
                        "non_null_count": int(df[column].count()),
                        "null_count": int(df[column].isnull().sum()),
                        "unique_values": int(df[column].nunique()),
                        "sample_values": sample_values,
                    }
                    schema_info["columns"].append(col_info)
                except Exception as e:
                    logger.error(f"Error processing column {column}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            schema_info["error"] = str(e)
            schema_info["missing_columns"] = True

        return schema_info

    def save_schema(self, schema_info: Dict[str, Any], output_path: Path) -> None:
        """
        Save schema information to a JSON file.

        Args:
            schema_info (Dict[str, Any]): Schema information to save
            output_path (Path): Path where to save the schema file
        """
        try:
            # Ensure the directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the schema file
            with open(output_path, "w") as f:
                json.dump(schema_info, f, indent=4, cls=NumpyEncoder)
            logger.info(f"Schema saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving schema to {output_path}: {str(e)}")
            raise

    def process_directory(self) -> None:
        """
        Process all supported files in the input directory and create schema files.
        """
        if self.input_path.is_file():
            files = [self.input_path]
        else:
            files = [
                f
                for f in self.input_path.rglob("*")
                if f.is_file() and self.is_supported_file(f)
            ]

        for file_path in files:
            logger.info(f"Processing {file_path}")
            schema_info = self.get_schema(file_path)

            # Create output path with .schema extension
            output_path = file_path.with_suffix(file_path.suffix + ".schema")
            self.save_schema(schema_info, output_path)
