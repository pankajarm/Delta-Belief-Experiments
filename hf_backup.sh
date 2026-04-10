#!/bin/bash
# HF backup for 8x A100 delta-belief-rl checkpoints
set -euo pipefail

HF_REPO="pankajmathur/delta-belief-rl-8xa100"
HF_TOKEN="${HF_TOKEN:?Set HF_TOKEN env var}"
CKPT_DIR="/home/ubuntu/delta-belief-rl/checkpoints/delta_belief_rl/train_8xa100_paper"

cd /home/ubuntu/delta-belief-rl
source .venv/bin/activate

for step_dir in $CKPT_DIR/global_step_*; do
    [ -d "$step_dir" ] || continue
    step_name=$(basename $step_dir)
    marker="$CKPT_DIR/.uploaded_$step_name"
    
    if [ -f "$marker" ]; then
        echo "Skip $step_name (already uploaded)"
        continue
    fi
    
    echo "Uploading $step_name..."
    python3 -c "
from huggingface_hub import HfApi
import os, sys
api = HfApi(token='$HF_TOKEN')
step = '$step_name'
step_dir = '$step_dir'
api.upload_folder(
    folder_path=step_dir,
    repo_id='$HF_REPO',
    path_in_repo=step,
    repo_type='model',
)
print(f'{step} uploaded!')
"
    touch "$marker"
    echo "$step_name backed up to HF"
done
echo "Backup complete"
