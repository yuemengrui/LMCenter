#!/bin/bash

# shellcheck disable=SC2164
cd /workspace/ModelWorker
nohup python manage_model_worker.py >/dev/null 2>&1 &
