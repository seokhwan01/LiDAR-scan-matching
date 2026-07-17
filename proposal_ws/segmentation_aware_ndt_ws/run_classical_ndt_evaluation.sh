#!/usr/bin/env bash

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${WORKSPACE_DIR}/evaluation/run_classical_ndt_evaluation.sh" "$@"
