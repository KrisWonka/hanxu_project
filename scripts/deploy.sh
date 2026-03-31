#!/usr/bin/env bash
#
# Deploy voice-agent to Orange Pi 5 via rsync + SSH.
# Usage: ./scripts/deploy.sh [user@host]
#
set -euo pipefail

TARGET="${1:-dev@orangepi.local}"
REMOTE_DIR="/home/dev/hanxu_project"

echo "=== Deploying to ${TARGET}:${REMOTE_DIR} ==="

echo "[1/4] Syncing project files..."
rsync -avz --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude '.DS_Store' \
  --exclude 'node_modules' \
  ./ "${TARGET}:${REMOTE_DIR}/"

echo "[2/4] Installing Python dependencies..."
ssh "${TARGET}" "cd ${REMOTE_DIR} && pip3 install --user -r requirements.txt"

echo "[3/4] Installing systemd service..."
ssh "${TARGET}" "sudo cp ${REMOTE_DIR}/deploy/voice-agent.service /etc/systemd/system/ && sudo systemctl daemon-reload"

echo "[4/4] Restarting service..."
ssh "${TARGET}" "sudo systemctl enable voice-agent && sudo systemctl restart voice-agent"

echo ""
echo "=== Deployment complete ==="
echo "Check status:  ssh ${TARGET} 'sudo systemctl status voice-agent'"
echo "View logs:     ssh ${TARGET} 'journalctl -u voice-agent -f'"
