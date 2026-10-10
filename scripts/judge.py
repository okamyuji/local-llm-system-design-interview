#!/usr/bin/env python3
"""rubric.mdの5観点で回答を採点した下書きを、Message Batches APIで作る。"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

HAND_TOTAL_RE = re.compile(r"^## Q([1-3])\b[^\n]*[:：]\s*(\d+)/50\s*$", re.M)
HAND_ITEM_RE = re.compile(r"^- 観点([1-5])[^:：\n]*[:：]\s*(\d+)点", re.M)
FENCE = "`" * 3  # Markdownのコードブロック記号。直書きすると計画書やdocsの中でブロックが閉じる
QUESTION_RE = re.compile(rf"^## Q([1-3]): ([^\n]+)\n.*?{FENCE}text\n(.*?)\n{FENCE}", re.M | re.S)
CRITERION_RE = re.compile(r"^## 観点([1-5]): ([^（\n]+)", re.M)


class JudgeError(Exception):
    pass


def _section(text: str, start: int) -> str:
    end = text.find("\n## ", start)
    return text[start:] if end == -1 else text[start:end]


def parse_hand_scores(text: str) -> dict[int, list[int]]:
    scores: dict[int, list[int]] = {}
    for head in HAND_TOTAL_RE.finditer(text):
        q, total = int(head.group(1)), int(head.group(2))
        items = HAND_ITEM_RE.findall(_section(text, head.end()))
        ids = [int(i) for i, _ in items]
        if ids != [1, 2, 3, 4, 5]:
            raise JudgeError(f"Q{q}: 観点1〜5が揃っていません: {ids}")
        values = [int(s) for _, s in items]
        if sum(values) != total:
            raise JudgeError(f"Q{q}: 観点の合計{sum(values)}が見出しの{total}と一致しません")
        scores[q] = values
    if not scores:
        raise JudgeError("採点の見出し（## Q<n> ...: <点>/50）がありません")
    return scores


def parse_questions(text: str) -> dict[int, tuple[str, str]]:
    found = {int(m.group(1)): (m.group(2).strip(), m.group(3).strip()) for m in QUESTION_RE.finditer(text)}
    if sorted(found) != [1, 2, 3]:
        raise JudgeError(f"questions.mdからQ1〜Q3を読み取れません: {sorted(found)}")
    return found


def parse_criteria_names(text: str) -> list[str]:
    found = {int(m.group(1)): m.group(2).strip() for m in CRITERION_RE.finditer(text)}
    if sorted(found) != [1, 2, 3, 4, 5]:
        raise JudgeError(f"rubric.mdから観点名1〜5を読み取れません: {sorted(found)}")
    return [found[i] for i in range(1, 6)]


MODEL_SHORT = {"claude-opus-5-5": "opus55", "claude-sonnet-5-5": "sonnet55"}
DIR_RE = re.compile(r"^[a-zA-Z0-9-]+$")
CUSTOM_ID_RE = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")
SPLIT_RE = re.compile(r"([a-zA-Z0-9-]+)__q([1-3])__([a-z0-9]+)__r(\d+)")


def make_custom_id(dir_name: str, q: int, model_short: str, run: int) -> str:
    # "__"を区切りに使うので、ディレクトリ名には下線を許さない
    if not DIR_RE.match(dir_name):
        raise JudgeError(f"ディレクトリ名に使えない文字があります（英数字とハイフンだけ）: {dir_name}")
    cid = f"{dir_name}__q{q}__{model_short}__r{run}"
    if not CUSTOM_ID_RE.match(cid):
        raise JudgeError(f"custom_idが64文字を超えます: {cid}")
    return cid


def split_custom_id(cid: str) -> tuple[str, int, str, int]:
    m = SPLIT_RE.fullmatch(cid)
    if not m:
        raise JudgeError(f"custom_idの形式が違います: {cid}")
    return m.group(1), int(m.group(2)), m.group(3), int(m.group(4))


MAX_TOKENS = 4096
SCHEMA = {
    "type": "object",
    "properties": {
        "truncated": {"type": "boolean"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "integer"},
                    "evidence": {"type": "array", "items": {"type": "string"}},
                    "reason": {"type": "string"},
                },
                "required": ["id", "score", "evidence", "reason"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["truncated", "criteria"],
    "additionalProperties": False,
}
RULES = """あなたはシステム設計面接の採点者です。上の採点基準だけを使い、回答を観点1〜5で採点してください。
- 観点ごとに、回答から根拠となる箇所を一字一句そのまま引用し、evidenceに入れてください。要約や言い換えは引用にしないでください。
- 点数は各観点のレンジの記述に機械的に当てはめ、印象点を加えないでください。
- reasonには、どのレンジに当てはめたかと、その理由を日本語1〜2文で書いてください。
- 回答が途中で切れている場合はtruncatedをtrueにし、書かれた範囲で採点してください。"""


def build_request(custom_id: str, model: str, rubric: str, question: str, answer: str, examples: str = "") -> dict:
    system = [{"type": "text", "text": rubric, "cache_control": {"type": "ephemeral"}}]
    if examples:
        system.append({"type": "text", "text": examples, "cache_control": {"type": "ephemeral"}})
    system.append({"type": "text", "text": RULES})
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": system,
            "messages": [{"role": "user", "content": f"出題:\n{question}\n\n<answer>\n{answer}\n</answer>"}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        },
    }


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


def normalize_quote(text: str) -> str:
    # 判定モデルはMarkdownの強調やコード記号を外して引用し、引用全体を「」で囲むことがある
    return normalize_ws(text.replace("**", "").replace("`", "")).strip("「」")


def check_schema(obj) -> str | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("criteria"), list):
        return "criteriaがありません"
    criteria = obj["criteria"]
    ids = [c.get("id") if isinstance(c, dict) else None for c in criteria]
    if len(criteria) != 5 or set(ids) != {1, 2, 3, 4, 5}:
        return f"観点の重複か欠落があります: {ids}"
    for c in criteria:
        score = c.get("score")
        # bool は int の派生型なので type で厳密に比べる
        if type(score) is not int or not 0 <= score <= 10:
            return f"観点{c['id']}の点数が0〜10の整数ではありません: {score!r}"
    return None


def find_fabricated(obj: dict, answer: str) -> list[str]:
    haystack = normalize_quote(answer)
    quotes = [q for c in obj["criteria"] for q in c.get("evidence", [])]
    return [q for q in quotes if normalize_quote(q) and normalize_quote(q) not in haystack]


def validate_judgment(text: str, answer: str) -> tuple[dict | None, str | None, list[str]]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None, "JSONとして読めません", []
    error = check_schema(obj)
    if error:
        return None, error, []
    fabricated = find_fabricated(obj, answer)
    if fabricated:
        return None, "引用が回答本文にありません", fabricated
    return obj, None, []


PASS_BAND = 0.8
PASS_MAE = 1.5
PASS_TOTAL = 20 / 24
PASS_STABLE = 0.9
MAX_BIAS = 1.5
TOTAL_TOLERANCE = 5
SPAN_TOLERANCE = 2


def band(score: int) -> int:
    return 0 if score <= 3 else 1 if score <= 7 else 2


def compare(hand: dict[tuple[str, int], list[int]], judged: dict[tuple[str, int], list[int]]) -> dict:
    pairs = [(h, j) for key, values in hand.items() for h, j in zip(values, judged[key])]
    n = len(pairs)
    return {
        "band_agree": sum(band(h) == band(j) for h, j in pairs) / n,
        "mae": sum(abs(h - j) for h, j in pairs) / n,
        "bias": sum(j - h for h, j in pairs) / n,
        "total_within": sum(abs(sum(hand[k]) - sum(judged[k])) <= TOTAL_TOLERANCE for k in hand) / len(hand),
    }


def stability(runs: list[dict[tuple[str, int], list[int]]]) -> float:
    spans = [max(v) - min(v) for key in runs[0] for v in zip(*(run[key] for run in runs))]
    return sum(s <= SPAN_TOLERANCE for s in spans) / len(spans)


def verdict(m: dict, stable: float, fabricated: int) -> list[str]:
    reasons = []
    if m["band_agree"] < PASS_BAND:
        reasons.append(f"レンジ一致が{PASS_BAND:.0%}未満")
    if m["mae"] > PASS_MAE:
        reasons.append(f"平均絶対誤差が{PASS_MAE}点超")
    if m["total_within"] < PASS_TOTAL:
        reasons.append(f"合計差{TOTAL_TOLERANCE}点以内の回答が{PASS_TOTAL:.0%}未満")
    if stable < PASS_STABLE:
        reasons.append(f"採点ごとの幅2点以内が{PASS_STABLE:.0%}未満")
    if fabricated:
        reasons.append(f"引用の捏造が{fabricated}件")
    if abs(m["bias"]) > MAX_BIAS:
        reasons.append(f"偏りが±{MAX_BIAS}点超")
    return reasons


API = "https://api.anthropic.com/v1/messages/batches"
API_VERSION = "2023-06-01"
BATCH_ID_RE = re.compile(r"^msgbatch_[A-Za-z0-9]+$")


def http_request(method: str, url: str, key: str, body: dict | None = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"x-api-key": key, "anthropic-version": API_VERSION, "content-type": "application/json"}
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return res.read()
    except urllib.error.HTTPError as e:
        raise JudgeError(f"HTTP {e.code}: {e.read().decode(errors='replace')}") from e
    except urllib.error.URLError as e:
        raise JudgeError(f"APIに接続できません: {e.reason}") from e


def create_batch(requests: list[dict], key: str, http=http_request) -> str:
    return json.loads(http("POST", API, key, {"requests": requests}))["id"]


def check_batch_id(batch_id: str) -> str:
    # batch ID は URL とディレクトリ名に使うので、形式外は通さない
    if not BATCH_ID_RE.match(batch_id):
        raise JudgeError(f"batch IDの形式が違います: {batch_id!r}")
    return batch_id


def get_batch(batch_id: str, key: str, http=http_request) -> dict:
    return json.loads(http("GET", f"{API}/{check_batch_id(batch_id)}", key))


def get_results(url: str, key: str, http=http_request) -> list[dict]:
    return [json.loads(line) for line in http("GET", url, key).decode().splitlines() if line.strip()]


def classify(result: dict, answers: dict[tuple[str, int], str]) -> dict:
    cid = result["custom_id"]
    dir_name, q, model_short, run = split_custom_id(cid)
    rec = {"custom_id": cid, "dir": dir_name, "q": q, "model": model_short, "run": run}
    outcome = result["result"]
    if outcome["type"] != "succeeded":
        return {**rec, "status": outcome["type"]}
    text = "".join(b.get("text", "") for b in outcome["message"]["content"] if b.get("type") == "text")
    judgment, error, fabricated = validate_judgment(text, answers[(dir_name, q)])
    if error:
        return {**rec, "status": "invalid", "error": error, "fabricated": fabricated, "raw": text}
    return {**rec, "status": "ok", "judgment": judgment}


def render_scoring(dir_name: str, model: str, batch_id: str, titles: dict[int, str],
                   names: list[str], records: dict[int, dict]) -> str:
    lines = [f"# {dir_name} 自動採点の下書き", "",
             f"判定モデルは{model}、batch IDは{batch_id}です。人が回答と照合してから`scoring.md`へ反映してください。", ""]
    totals = []
    for q in (1, 2, 3):
        rec = records.get(q, {"status": "missing"})
        if rec["status"] != "ok":
            lines += [f"## Q{q} {titles[q]}: 判定なし（{rec['status']}）", ""]
            continue
        criteria = sorted(rec["judgment"]["criteria"], key=lambda c: c["id"])
        totals.append(sum(c["score"] for c in criteria))
        lines += [f"## Q{q} {titles[q]}: {totals[-1]}/50", ""]
        if rec["judgment"]["truncated"]:
            lines += ["回答は途中で切れていると判定しました。", ""]
        for c in criteria:
            quotes = "".join(f"「{e}」" for e in c["evidence"])
            lines.append(f"- 観点{c['id']}・{names[c['id'] - 1]}: {c['score']}点。{c['reason']}{quotes}")
        lines.append("")
    if len(totals) == 3:
        lines.append(f"- 合計: {sum(totals)}/150")
    return "\n".join(lines).rstrip() + "\n"


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = "claude-sonnet-5-5"
SHORT_MODEL = {v: k for k, v in MODEL_SHORT.items()}


def list_dirs(root: Path) -> list[str]:
    return sorted(p.parent.name for p in (root / "results").glob("*/q1_raw.txt"))


def load_answers(root: Path, dirs: list[str]) -> dict[tuple[str, int], str]:
    return {(d, q): (root / "results" / d / f"q{q}_raw.txt").read_text() for d in dirs for q in (1, 2, 3)}


def scores_of(rec: dict) -> list[int]:
    return [c["score"] for c in sorted(rec["judgment"]["criteria"], key=lambda c: c["id"])]


def read_hand_scoring(root: Path, dir_name: str) -> str:
    path = root / "results" / dir_name / "scoring.md"
    if not path.exists():
        raise JudgeError(f"手採点がありません: {path}")
    return path.read_text()


def pick_examples(root: Path, dirs: list[str], chosen: list[str] | None) -> list[str]:
    if chosen is None:
        return [d for d in list_dirs(root) if d not in dirs and (root / "results" / d / "scoring.md").exists()]
    overlap = sorted(set(chosen) & set(dirs))
    if overlap:
        raise JudgeError(f"採点対象を採点例に含めることはできません: {overlap}")
    return chosen


def load_examples(root: Path, dirs: list[str]) -> str:
    if not dirs:
        return ""
    blocks = [f'<example dir="{d}">\n{read_hand_scoring(root, d)}\n</example>' for d in dirs]
    return "次は手採点の採点例です。採点の水準をこの例に合わせてください。\n\n" + "\n\n".join(blocks)


def cmd_submit(args, root: Path, key: str, http) -> int:
    rubric = (root / "rubric.md").read_text()
    questions = parse_questions((root / "questions.md").read_text())
    dirs = args.dirs or list_dirs(root)
    models = args.model or [DEFAULT_MODEL]
    ids = [(make_custom_id(d, q, MODEL_SHORT[m], r), m, d, q)
           for m in models for r in range(args.runs) for d in dirs for q in (1, 2, 3)]
    answers = load_answers(root, dirs)
    examples = load_examples(root, pick_examples(root, dirs, args.examples))
    requests = [build_request(cid, m, rubric, questions[q][1], answers[(d, q)], examples) for cid, m, d, q in ids]
    batch_id = create_batch(requests, key, http)
    print(f"{batch_id} を作成しました（{len(requests)}件）。終わったら collect {batch_id} を実行してください。")
    return 0


def cmd_collect(args, root: Path, key: str, http) -> int:
    batch = get_batch(args.batch_id, key, http)
    if batch["processing_status"] != "ended":
        print(f"処理中です（{batch['processing_status']}）: {batch['request_counts']}")
        return 2
    results = get_results(batch["results_url"], key, http)
    dirs = sorted({split_custom_id(r["custom_id"])[0] for r in results})
    answers = load_answers(root, dirs)
    records = [classify(r, answers) for r in results]
    out = root / "judge-out" / args.batch_id
    out.mkdir(parents=True, exist_ok=True)
    for rec in records:
        (out / f"{rec['custom_id']}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    counts = Counter(rec["status"] for rec in records)
    print(" / ".join(f"{k} {v}" for k, v in sorted(counts.items())) + f"（保存先 {out}）")
    if len({(r["model"], r["run"]) for r in records}) == 1:
        write_drafts(root, args.batch_id, records)
    return 0


def write_drafts(root: Path, batch_id: str, records: list[dict]) -> None:
    titles = {q: t for q, (t, _) in parse_questions((root / "questions.md").read_text()).items()}
    names = parse_criteria_names((root / "rubric.md").read_text())
    model = SHORT_MODEL[records[0]["model"]]
    for d in sorted({r["dir"] for r in records}):
        by_q = {r["q"]: r for r in records if r["dir"] == d}
        path = root / "results" / d / "scoring.judge.md"
        path.write_text(render_scoring(d, model, batch_id, titles, names, by_q))
        print(f"下書きを書きました: {path}")


def load_hand(root: Path, dirs: list[str]) -> dict[tuple[str, int], list[int]]:
    return {(d, q): s for d in dirs for q, s in parse_hand_scores(read_hand_scoring(root, d)).items()}


def cmd_calibrate(args, root: Path) -> int:
    folder = root / "judge-out" / check_batch_id(args.batch_id)
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    if not records:
        raise JudgeError(f"{folder}に判定がありません。先にcollectを実行してください")
    hand = load_hand(root, sorted({r["dir"] for r in records}))
    for model in sorted({r["model"] for r in records}):
        print(calibrate_model(model, hand, [r for r in records if r["model"] == model]))
    return 0


def calibrate_model(model: str, hand: dict, records: list[dict]) -> str:
    runs: dict[int, dict[tuple[str, int], list[int]]] = {r["run"]: {} for r in records}
    for r in records:
        if r["status"] == "ok":
            runs[r["run"]][(r["dir"], r["q"])] = scores_of(r)
    missing = sum(1 for run in runs.values() for k in hand if k not in run)
    if missing or 0 not in runs:
        return f"{model}: 判定不能（判定が欠けた回答 {missing}件）"
    m = compare(hand, runs[0])
    stable = stability([runs[k] for k in sorted(runs)])
    fabricated = sum(1 for r in records if r.get("fabricated"))
    reasons = verdict(m, stable, fabricated)
    head = "合格" if not reasons else "不合格（" + "、".join(reasons) + "）"
    return (f"{model}: {head}\n"
            f"  レンジ一致 {m['band_agree']:.1%} / 平均絶対誤差 {m['mae']:.2f} / 偏り {m['bias']:+.2f}"
            f" / 合計差5点以内 {m['total_within']:.1%} / {len(runs)}回の幅2点以内 {stable:.1%} / 捏造 {fabricated}件")


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("1以上を指定してください")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rubric.mdで回答を採点した下書きを、Message Batches APIで作ります。")
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit", help="回答をまとめて1つのバッチに送る")
    s.add_argument("--model", action="append", choices=sorted(MODEL_SHORT), help=f"判定モデル（既定 {DEFAULT_MODEL}、複数指定可）")
    s.add_argument("--examples", action="append", help="採点例にする手採点済みディレクトリ（省略時は対象以外の手採点すべて、複数指定可）")
    s.add_argument("--runs", type=positive_int, default=1, help="同じ回答を採点する回数")
    s.add_argument("dirs", nargs="*", help="results/配下の対象ディレクトリ（省略時は全部）")
    for name, text in (("collect", "終わったバッチの結果を保存し、下書きを書く"), ("calibrate", "手採点と比べて合否を出す")):
        sub.add_parser(name, help=text).add_argument("batch_id")
    return parser


def main(argv: list[str] | None = None, root: Path = ROOT, http=http_request, env=os.environ) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "calibrate":
            return cmd_calibrate(args, root)
        key = env.get("ANTHROPIC_API_KEY")
        if not key:
            raise JudgeError("環境変数ANTHROPIC_API_KEYを設定してください")
        return (cmd_submit if args.cmd == "submit" else cmd_collect)(args, root, key, http)
    except JudgeError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
