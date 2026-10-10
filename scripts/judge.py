#!/usr/bin/env python3
"""rubric.mdの5観点で回答を採点した下書きを、Message Batches APIで作る。"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

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


def build_request(custom_id: str, model: str, rubric: str, question: str, answer: str) -> dict:
    return {
        "custom_id": custom_id,
        "params": {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": [
                {"type": "text", "text": rubric, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": RULES},
            ],
            "messages": [{"role": "user", "content": f"出題:\n{question}\n\n<answer>\n{answer}\n</answer>"}],
            "output_config": {"format": {"type": "json_schema", "schema": SCHEMA}},
        },
    }


def normalize_ws(text: str) -> str:
    return " ".join(text.split())


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
    haystack = normalize_ws(answer)
    quotes = [q for c in obj["criteria"] for q in c.get("evidence", [])]
    return [q for q in quotes if normalize_ws(q) and normalize_ws(q) not in haystack]


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
        reasons.append("合計差5点以内の回答が24回答中20未満")
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
