#!/bin/bash
# 複数のGGUFモデルを順にllama-serverで起動し、各3問を実測する
#
# 使い方: MODELDIR=... RESULTS=... ./run_all_models.sh
# 環境変数:
#   MODELDIR     	GGUFファイルの置き場所 (必須)
#   RESULTS      	結果の出力先 (default: ./results)
#   LLAMA_SERVER 	llama-serverのパス (default: llama-server)
#   PORT         	実験用サーバーのポート (default: 8081)
#   AGENT_BIN, AGENT_WS はrun_model.shへ引き継ぐ
set -u
MODELDIR="${MODELDIR:?GGUFの置き場所をMODELDIRで指定してください}"
RESULTS="${RESULTS:-$PWD/results}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
PORT="${PORT:-8081}"
RUNNER="$(cd "$(dirname "$0")" && pwd)/run_model.sh"

run_one() {
  local gguf="$1" outdir="$2" alias="$3"
  # 共有の /tmp に固定名で書くと、他の利用者が置いた symlink を上書きしうる
  local log
  local tmp="${TMPDIR:-/tmp}"
  log="$(mktemp "${tmp%/}/llama-$PORT-$outdir.XXXXXX")" || return 1
  echo "##### $outdir: server start $(date +%H:%M:%S) log=$log"
  "$LLAMA_SERVER" -m "$MODELDIR/$gguf" --jinja -c 8192 --port "$PORT" --host 127.0.0.1 \
    > "$log" 2>&1 &
  local pid=$!
  local ok=0
  for _ in $(seq 1 60); do
    if curl -s --max-time 2 "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then ok=1; break; fi
    sleep 3
  done
  if [ "$ok" != 1 ]; then
    echo "##### $outdir: SERVER FAILED TO START"; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 1
  fi
  echo "##### $outdir: server healthy, running questions"
  bash "$RUNNER" "$RESULTS/$outdir" "llamacpp/$alias" experiment-config.yaml
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  echo "##### $outdir: server stopped $(date +%H:%M:%S)"
}

# 計測対象。GGUFファイル名 / 出力ディレクトリ名 / configのallow_modelsに載せた別名
run_one google_gemma-4-E4B-it-Q4_K_M.gguf gemma4-e4b gemma
run_one Qwen3.5-9B-Q4_K_M.gguf qwen35-9b qwen35
run_one Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf qwen25-coder-14b qwencoder
echo ALL_MODELS_DONE
