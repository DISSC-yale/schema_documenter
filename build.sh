#!/bin/bash

# Clean previous builds
rm -rf dist/ build/ *.egg-info

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install build dependencies with uv
echo "Installing build dependencies..."
uv pip install build twine

# Build the package
echo "Building package..."
python -m build

echo "Build complete! Distribution files are in the 'dist' directory."
echo
echo "To publish to PyPI, run:"
echo "  python -m twine upload dist/*"
echo
echo "To install locally, run:"
echo "  uv pip install dist/*.whl" 