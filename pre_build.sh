#!/bin/bash
set -e

echo "=== 🧠 Checking SAM model in persistent storage (/data/models) ==="

MODEL_DIR="/data/models"
SAM_MODEL="${MODEL_DIR}/sam_vit_b_01ec64.pth"

# Ensure directory exists (Render persistent disk)
mkdir -p "$MODEL_DIR"

# Just check if it’s there
if [ -f "$SAM_MODEL" ] && [ -s "$SAM_MODEL" ]; then
    echo "✅ SAM model already exists:"
    ls -lh "$SAM_MODEL"
else
    echo "❌ SAM model missing in /data/models!"
    echo "Please upload manually using /upload_model endpoint after first deploy."
    ls -lh "$MODEL_DIR" || true
fi

echo "=== ✅ Pre-build check complete ==="
