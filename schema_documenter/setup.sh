#!/bin/bash

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
echo "Creating virtual environment..."
python -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
echo "Installing dependencies..."
make install-dev

echo "Setup complete! You can now run:"
echo "  make test    # Run tests"
echo "  make lint    # Run linters"
echo "  make format  # Format code" 