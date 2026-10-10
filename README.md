# local-llm-system-design-interview

ローカルLLMにシステム設計面接の問題を解かせ、固定の採点基準で採点する実測リポジトリです。出題、採点基準、実行スクリプト、各モデルの回答全文と採点根拠をすべて収録しているので、読者が自分のマシンで同じ計測を再現できます。

## 収録内容

| パス | 内容 |
| --- | --- |
| `questions.md` | 出題3問（チケット予約、画像アップロード、リアルタイムチャット）と実行条件 |
| `rubric.md` | 採点基準。実測前に固定した5基準×10点、合計50点 |
| `scripts/run_model.sh` | 1モデルに3問を出題し、回答と所要時間を保存するスクリプト |
| `scripts/run_all_models.sh` | 複数のGGUFをllama-serverで順に起動して一括計測するスクリプト |
| `scripts/run_model_tools.sh` | web_searchとweb_fetchの利用を認めた検索併用条件のスクリプト |
| `results/<モデル名>/q*_raw.txt` | 各問への回答全文（無加工） |
| `results/<モデル名>/q*_time.txt` | 各問の所要時間（`/usr/bin/time -p`の出力） |
| `results/<モデル名>/scoring.md` | 採点結果と、観点ごとの根拠 |
| `results/<モデル名>-tools/` | 検索併用条件（web_search / web_fetch許可）の回答と採点 |

## 前提

計測には次の3つを使います。

- [llama.cpp](https://github.com/ggml-org/llama.cpp)のllama-server（Homebrewなら`brew install llama.cpp`）
- [go-llm-agent](https://github.com/okamyuji/go-llm-agent)の`agent`バイナリ（1ショット実行の`agent run`を使用）
- 計測したいモデルのGGUFファイル（Hugging Faceから取得）

llama-serverを使うのは、OpenAI互換APIで複数のGGUFを同じ条件で差し替えられるためです。goで書かれたagentを経由するのは、実際のエージェント利用と同じsystem promptとパラメータ（max_tokens 2048、repeat_penalty 1.15）で測るためです。

## 使い方

### 1. モデルを用意する

計測対象のGGUFを1つのディレクトリへ置きます。この実測で使ったモデルは次の4つです。

- [Shisa-v2 Mistral-Nemo 12B (Q4_K_M)](https://huggingface.co/mradermacher/Shisa-v2-Mistral-Nemo-12B-Abliterated-i1-GGUF)
- [Gemma 4 E4B instruct (Q4_K_M)](https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF)
- [Qwen3.5 9B (Q4_K_M)](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF)
- [Qwen2.5 Coder 14B instruct (Q4_K_M)](https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF)

### 2. agentの設定を用意する

go-llm-agentのワークスペースに、llama-serverのポートを向いたproviderを持つconfigを置きます。既定の`config.yaml`との差分は`base_url`のポートと`allow_models`だけです。

```yaml
providers:
  llamacpp:
    base_url: http://127.0.0.1:8081/v1
    allow_models: [gemma, qwen35, qwencoder]
```

`allow_models`の名前はagentへ渡す別名で、llama-server側は起動時に`-m`で渡した1モデルだけを提供するため、任意の名前で構いません。

### 3. 計測する

```bash
MODELDIR=/path/to/gguf \
RESULTS=$PWD/results \
AGENT_BIN=/path/to/go-llm-agent/bin/agent \
AGENT_WS=/path/to/go-llm-agent/llm-agent-workspace \
bash scripts/run_all_models.sh
```

モデルごとにllama-serverをポート8081で起動し、`/health`の応答を待ってから3問を順に出題し、終わったらサーバーを止めて次のモデルへ進む流れです。回答は`results/<モデル名>/q1_raw.txt`〜`q3_raw.txt`へ、所要時間は`q*_time.txt`へ保存されます。

1モデルだけ測る場合は、llama-serverを自分で起動したうえで`run_model.sh`を直接呼んでください。

```bash
AGENT_BIN=... AGENT_WS=... bash scripts/run_model.sh results/my-model llamacpp/gemma experiment-config.yaml
```

### 4. 採点する

`rubric.md`の5基準へ機械的に当てはめます。採点は回答を見る前に固定した基準で行い、基準ごとに回答から根拠を引用して`results/<モデル名>/scoring.md`へ残します。

### 5. 自動採点の下書きを作る（任意）

`scripts/judge.py`は、`rubric.md`の5観点で回答を採点した下書きを、AnthropicのMessage Batches APIで作ります。下書きは`results/<モデル名>/scoring.judge.md`に書かれ、`scoring.md`は変更しません。下書きは必ず人が回答と照合してから`scoring.md`へ反映してください。

```bash
export ANTHROPIC_API_KEY=...            # Claude ConsoleのAPIキー
python3 scripts/judge.py submit my-model # 対象ディレクトリを省くとresults/配下すべて
python3 scripts/judge.py collect <batch ID>
```

バッチの処理には最長24時間かかります。`collect`は処理中なら状況を表示して終了コード2で終わるので、時間をおいて再実行してください。判定の生データは`judge-out/<batch ID>/`に保存されます。

採点の水準を手採点に合わせるため、手採点済みのほかのモデルの`scoring.md`を採点例として渡します。`--examples`を省くと、採点対象以外の手採点がすべて採点例になります。採点対象と採点例には`results/`配下のディレクトリ名をそのまま書き、採点対象を採点例に含めることはできません。採点例が1つもないときは、校正していない構成なので警告を出します。

手採点と判定の一致を確かめるときは、手採点済みの回答を採点例と採点対象に分け、同じ回答を3回採点してから`calibrate`で比べてください。`calibrate`は、引用の捏造が1件でもあれば不合格とし、採点が3回に満たないときは判定しません。

```bash
python3 scripts/judge.py submit --model claude-opus-5-5 --runs 3 \
  --examples gemma4-e4b --examples gemma4-e4b-tools --examples qwen35-9b --examples qwen35-9b-tools \
  qwen25-coder-14b qwen25-coder-14b-tools shisa-v2-12b shisa-v2-12b-tools
python3 scripts/judge.py collect <batch ID>
python3 scripts/judge.py calibrate <batch ID>
```

## 計測条件の注意

- 各問は独立したプロセスの1ショット実行で、会話履歴を持ち越しません
- max_tokens 2048で切れた回答は、切れたという事実ごと記録します（継続要求は送りません）
- 出題文の「日本語の文章だけで回答してください」は、日本語指定なしだと回答全体が英語になるモデルがあったため追加した条件です（`results/shisa-v2-12b/q1_pilot_english.txt`が指定なしの観測記録です）
- 量子化はすべてQ4_K_Mに揃えていますが、量子化の違いは回答品質に影響し得ます

## ライセンス

MITライセンスです。詳細は[LICENSE](LICENSE)を参照してください。
