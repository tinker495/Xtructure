#!/bin/bash
# Development setup script for xtructure

echo "Setting up development environment..."

# Create the project environment and install development dependencies.
uv sync --group dev

echo "✅ Development environment ready!"
