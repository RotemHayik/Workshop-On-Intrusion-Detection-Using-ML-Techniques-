param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskModel = Join-Path $taskRoot '05_models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf'
$taskRuntime = Join-Path $taskRoot '05_models/llm/runtime/llama-server.exe'
& $taskRuntime -m $taskModel --host 127.0.0.1 --port $Port -c 2048 -np 1 -t 2 -tb 2 -n 160
