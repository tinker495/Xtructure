#!/bin/bash
# Development setup script for xtructure

echo "Setting up development environment..."

# Uninstall existing installation
pip uninstall xtructure -y 2>/dev/null || true

# Install in editable mode with development dependencies
pip install -e ".[dev]"

echo "✅ Development environment ready!"