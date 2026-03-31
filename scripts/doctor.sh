#!/usr/bin/env bash

set -euo pipefail

echo "== Orange Pi Dev Doctor =="
echo

if ! command -v docker >/dev/null 2>&1; then
  echo "[FAIL] docker command not found."
  echo "Install Docker Desktop first."
  exit 1
fi

echo "[OK] docker CLI: $(docker --version)"

if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
  echo "[OK] compose mode: docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
  echo "[OK] compose mode: docker-compose"
else
  echo "[FAIL] Neither 'docker compose' nor 'docker-compose' is available."
  echo "Try: brew install docker-compose"
  exit 1
fi

if docker info >/dev/null 2>&1; then
  echo "[OK] Docker daemon is running."
else
  echo "[WARN] Docker daemon is not running yet."
  if command -v colima >/dev/null 2>&1; then
    echo "Try: colima start --cpu 4 --memory 8 --disk 60"
  else
    echo "Open Docker Desktop and wait until it says Engine running,"
    echo "or install Colima: brew install colima && colima start"
  fi
  exit 1
fi

IMAGE="hanxu_project-dev"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[OK] Dev image '$IMAGE' exists."
else
  echo "[INFO] Dev image '$IMAGE' not built yet. Run: $COMPOSE_CMD build"
fi

echo
echo "[OK] Environment is ready."
echo "Next:"
echo "  $COMPOSE_CMD build"
echo "  $COMPOSE_CMD run --rm dev"
