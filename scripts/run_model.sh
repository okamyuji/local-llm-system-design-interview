#!/bin/bash
# 1モデルに3問を順に出題し、回答と所要時間を保存する
#
# 使い方: run_model.sh <出力ディレクトリ> <provider/model> [configファイル名]
# 環境変数:
#   AGENT_BIN 	go-llm-agentのagentバイナリ (default: agent)
#   AGENT_WS  	agentのワークスペースディレクトリ (config相対パスの基準)
set -u
OUTDIR="$1"
MODEL="$2"
CONFIG="${3:-config.yaml}"
AGENT_BIN="${AGENT_BIN:-agent}"
AGENT_WS="${AGENT_WS:-$PWD}"
mkdir -p "$OUTDIR"
cd "$AGENT_WS" || exit 1

Q1='あなたはシステム設計面接の候補者です。ツールを使わず、日本語の文章だけで回答してください。

問題: 人気イベントのチケット予約サービスを設計してください。座席指定のあるイベントを扱い、発売開始直後に大量のアクセスが集中します。設計の考え方を、重要だと思う順に説明してください。'

Q2='あなたはシステム設計面接の候補者です。ツールを使わず、日本語の文章だけで回答してください。

問題: 利用者が画像をアップロードして共有できるサービスを設計してください。アップロードされた画像は加工処理を経てから他の利用者に公開されます。設計の考え方を、重要だと思う順に説明してください。'

Q3='あなたはシステム設計面接の候補者です。ツールを使わず、日本語の文章だけで回答してください。

問題: 1対1とグループの両方に対応するリアルタイムチャットサービスを設計してください。オフラインだった利用者が復帰したときのメッセージの扱いも含めて、設計の考え方を重要だと思う順に説明してください。'

i=1
for Q in "$Q1" "$Q2" "$Q3"; do
  echo "=== Q$i start $(date +%H:%M:%S)"
  /usr/bin/time -p "$AGENT_BIN" run -config "$CONFIG" -model "$MODEL" -p "$Q" \
    > "$OUTDIR/q${i}_raw.txt" 2> "$OUTDIR/q${i}_time.txt"
  echo "=== Q$i done rc=$? $(tail -3 "$OUTDIR/q${i}_time.txt" | head -1) bytes=$(wc -c < "$OUTDIR/q${i}_raw.txt")"
  i=$((i+1))
done
echo ALL_DONE
