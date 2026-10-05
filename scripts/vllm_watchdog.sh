#!/bin/bash
# Relaunch vLLM if no vLLM server process exists for 3 consecutive checks (90s).
# Never touches a running (even slow-starting) server. Log: /opt/vllm-watchdog.log
misses=0
while true; do
  if pgrep -f "bin/vllm serve" >/dev/null; then
    misses=0
  else
    misses=$((misses+1))
    if [ $misses -ge 3 ]; then
      echo "$(date '+%F %T') no vLLM process; relaunching" >> /opt/vllm-watchdog.log
      [ -f /opt/vllm-serve.log ] && mv /opt/vllm-serve.log /opt/vllm-serve.$(date +%H%M%S).log
      cd /workspace && PATH=/opt/vllm-venv/bin:$PATH nohup /opt/vllm-venv/bin/vllm serve /workspace/models/gemma-4-31b-it-qat-w4a16-ct \
        --served-model-name google/gemma-4-31b-it-qat-w4a16-ct --enable-auto-tool-choice --tool-call-parser gemma4 \
        --reasoning-parser gemma4 --max-model-len 32768 --kv-cache-memory 23000000000 > /opt/vllm-serve.log 2>&1 &
      misses=0
      sleep 120
    fi
  fi
  sleep 30
done
