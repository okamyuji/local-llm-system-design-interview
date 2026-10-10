#!/usr/bin/env python3
"""rubric.mdの5観点で回答を採点した下書きを、Message Batches APIで作る。"""
from __future__ import annotations

import re

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
