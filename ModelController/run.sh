#!/bin/bash

# shellcheck disable=SC2164
cd /workspace/ModelController
nohup python manage_model_controller.py >/dev/null 2>&1 &
