#!/bin/bash
# Hapa LLaDA Node Installer

echo "Initializing Hapa LLaDA Node environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

chmod +x start.sh
chmod +x hapa-llada-node
echo "Installation complete. Run ./start.sh to launch."
