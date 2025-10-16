#!/bin/bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Download SAM checkpoint if missing
SAM_MODEL="sam_vit_b_01ec64.pth"
if [ ! -f "$SAM_MODEL" ]; then
    echo "Downloading SAM checkpoint..."
    curl -L -o "$SAM_MODEL" "https://huggingface.co/facebook/segment-anything/resolve/main/sam_vit_b_01ec64.pth"
fi
