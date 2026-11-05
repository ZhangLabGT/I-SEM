#!/usr/bin/env bash
# One queue pass over the fits; safe to run several at once (claim files).
W=${1:-1}
cd "$(dirname "$0")"
bash run.sh harness.py fit --kind sims --block weak --arms wald no_push_signed --workers "$W" --spread
bash run.sh harness.py fit --kind sims --block contrast --arms shipped no_push no_push_signed no_se no_stability no_half wald --workers "$W" --spread
bash run.sh harness.py fit --kind real --arms wald no_push_signed --workers 1
