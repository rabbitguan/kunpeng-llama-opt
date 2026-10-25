#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
COMMAND=${1:-help}
shift || true

case "$COMMAND" in
    setup|baseline|local-test)
        exec python3 "$SCRIPT_DIR/scripts/experiment.py" "$COMMAND" --config "$SCRIPT_DIR/config.json" "$@"
        ;;
    *)
        echo "Usage: bash tmac_experiment/run.sh {setup|baseline|local-test} [--set dotted.path=value]" >&2
        exit 2
        ;;
esac
