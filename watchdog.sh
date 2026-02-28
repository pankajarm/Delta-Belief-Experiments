#!/bin/bash
# Watchdog: monitors training, auto-restarts on crash, saves checkpoints
# Run with: nohup bash watchdog.sh &

LOGFILE="/home/ubuntu/delta-belief-rl/.local_logs/watchdog.log"
TRAIN_LOG="/home/ubuntu/delta-belief-rl/.local_logs/nohup_train.log"
LAUNCH_SCRIPT="/home/ubuntu/delta-belief-rl/launch_train_8xa100_awq.sh"
HF_BACKUP="/home/ubuntu/delta-belief-rl/hf_backup.sh"
CHECK_INTERVAL=300  # 5 minutes

log() { echo "[$(date -u +%Y-%m-%d\ %H:%M:%S\ UTC)] $1" | tee -a "$LOGFILE"; }

log "=== Watchdog started ==="

while true; do
    # Check if training process is alive
    TRAIN_PID=$(pgrep -f "python3 -m train" | head -1)
    
    if [ -z "$TRAIN_PID" ]; then
        log "!!! TRAINING PROCESS DEAD - AUTO RESTARTING !!!"
        
        # Kill any orphan ray processes
        pkill -9 -f ray 2>/dev/null || true
        sleep 5
        
        # Check GPU is clear
        GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
        if [ "$GPU_MEM" -gt 1000 ]; then
            log "GPUs still occupied ($GPU_MEM MiB), waiting 30s..."
            sleep 30
            pkill -9 -f ray 2>/dev/null || true
            sleep 5
        fi
        
        # Run HF backup before restart (save any unsaved checkpoints)
        log "Running HF backup before restart..."
        bash "$HF_BACKUP" >> "$LOGFILE" 2>&1 || true
        
        # Restart training
        log "Launching training..."
        cd /home/ubuntu/delta-belief-rl
        > "$TRAIN_LOG"
        nohup bash "$LAUNCH_SCRIPT" >> "$TRAIN_LOG" 2>&1 &
        NEW_PID=$!
        log "Training launched with PID $NEW_PID"
        
        # Wait for model loading
        sleep 120
    else
        # Training alive - check for OOM signs in recent log
        if tail -50 "$TRAIN_LOG" 2>/dev/null | grep -qi "OOM\|out of memory\|CUDA error\|ActorDiedError\|SYSTEM_ERROR"; then
            log "!!! OOM/ERROR DETECTED - killing and restarting !!!"
            kill -9 $TRAIN_PID 2>/dev/null || true
            pkill -9 -f ray 2>/dev/null || true
            sleep 10
            continue  # Will trigger restart on next loop
        fi
        
        # Check GPU memory for potential OOM buildup
        MAX_GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        if [ "$MAX_GPU_MEM" -gt 79000 ]; then
            log "WARNING: GPU memory critically high ($MAX_GPU_MEM MiB / 81920 MiB)"
        fi
    fi
    
    # Run HF backup (uploads any new checkpoints)
    bash "$HF_BACKUP" >> "$LOGFILE" 2>&1 || true
    
    sleep $CHECK_INTERVAL
done
