#!/bin/bash
# 検索併用条件: web_search / web_fetch の利用を認めて3問を出題する
#
# 使い方: run_model_tools.sh <出力ディレクトリ> <provider/model> [configファイル名]
# 環境変数: AGENT_BIN / AGENT_WS はrun_model.shと同じ
set -u
OUTDIR="$1"
MODEL="$2"
CONFIG="${3:-config.yaml}"
AGENT_BIN="${AGENT_BIN:-agent}"
AGENT_WS="${AGENT_WS:-$PWD}"
mkdir -p "$OUTDIR"
cd "$AGENT_WS" || exit 1

P='あなたはシステム設計面接の候補者です。必要に応じてweb_searchやweb_fetchツールで情報を調べて構いません。最終的な回答は日本語の文章でまとめてください。

問題: '

Q1='人気イベントのチケット予約サービスを設計してください。座席指定のあるイベントを扱い、発売開始直後に大量のアクセスが集中します。設計の考え方を、重要だと思う順に説明してください。'
Q2='利用者が画像をアップロードして共有できるサービスを設計してください。アップロードされた画像は加工処理を経てから他の利用者に公開されます。設計の考え方を、重要だと思う順に説明してください。'
Q3='1対1とグループの両方に対応するリアルタイムチャットサービスを設計してください。オフラインだった利用者が復帰したときのメッセージの扱いも含めて、設計の考え方を重要だと思う順に説明してください。'

i=1
for Q in "$Q1" "$Q2" "$Q3"; do
  echo "=== Q$i start $(date +%H:%M:%S)"
  /usr/bin/time -p "$AGENT_BIN" run -config "$CONFIG" -model "$MODEL" -p "$P$Q" \
    > "$OUTDIR/q${i}_raw.txt" 2> "$OUTDIR/q${i}_time.txt"
  echo "=== Q$i done rc=$? $(grep -m1 '^real' "$OUTDIR/q${i}_time.txt") bytes=$(wc -c < "$OUTDIR/q${i}_raw.txt")"
  i=$((i+1))
done
echo ALL_DONE
