#!/usr/bin/env bash
set -euo pipefail

bash commands/train_baseline.sh
bash commands/train_temporal.sh
