#!/bin/bash
# Quick start script for CRS Dashboard

echo "=========================================="
echo "CRS Death Risk Assessment Dashboard"
echo "=========================================="
echo ""

# Check if model exists
if [ ! -f "crs_model_best.pkl" ]; then
    echo "⚠️  Warning: CRS model not found!"
    echo "   Please run first: python 12_crs_model_training.py"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found!"
    echo "   Install with: pip install streamlit"
    exit 1
fi

echo "✅ Starting dashboard..."
echo ""
echo "The dashboard will open in your browser at:"
echo "   http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run dashboard
streamlit run crs_dashboard.py

