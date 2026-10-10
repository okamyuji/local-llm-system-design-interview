import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import judge  # noqa: E402

SAMPLE_HAND = """# m 採点

## Q1 題1: 15/50

- 観点1・壊してはいけない条件: 1点。理由
- 観点2・構成要素の選定: 2点。理由
- 観点3・主要フローの具体化: 3点。理由
- 観点4・異常系の想定: 4点。理由
- 観点5・スケールの議論: 5点。理由

## 特記事項

- 合計: 15/150
"""


class ParseHandScoresTest(unittest.TestCase):
    def test_reads_all_24_published_answers(self):
        files = sorted((ROOT / "results").glob("*/scoring.md"))
        self.assertEqual(len(files), 8)
        for f in files:
            scores = judge.parse_hand_scores(f.read_text())
            self.assertEqual(sorted(scores), [1, 2, 3], f)
            self.assertTrue(all(len(v) == 5 for v in scores.values()), f)

    def test_returns_scores_in_criterion_order(self):
        self.assertEqual(judge.parse_hand_scores(SAMPLE_HAND), {1: [1, 2, 3, 4, 5]})

    def test_raises_when_a_criterion_is_missing(self):
        text = SAMPLE_HAND.replace("- 観点5・スケールの議論: 5点。理由\n", "").replace("15/50", "10/50")
        with self.assertRaisesRegex(judge.JudgeError, "観点1〜5"):
            judge.parse_hand_scores(text)

    def test_raises_when_sum_differs_from_heading(self):
        with self.assertRaisesRegex(judge.JudgeError, "一致しません"):
            judge.parse_hand_scores(SAMPLE_HAND.replace("15/50", "16/50"))

    def test_raises_when_no_heading(self):
        with self.assertRaisesRegex(judge.JudgeError, "見出し"):
            judge.parse_hand_scores("# 空\n")


class ParseQuestionsTest(unittest.TestCase):
    def test_reads_three_questions_from_repo(self):
        qs = judge.parse_questions((ROOT / "questions.md").read_text())
        self.assertEqual(sorted(qs), [1, 2, 3])
        self.assertEqual(qs[1][0], "チケット予約サービス")
        self.assertTrue(qs[3][1].startswith("あなたはシステム設計面接の候補者です。"))
        self.assertIn("リアルタイムチャット", qs[3][1])

    def test_raises_when_a_question_is_missing(self):
        with self.assertRaisesRegex(judge.JudgeError, "Q1〜Q3"):
            judge.parse_questions(f"## Q1: 題\n\n{judge.FENCE}text\n本文\n{judge.FENCE}\n")


class ParseCriteriaNamesTest(unittest.TestCase):
    def test_reads_five_names_from_repo(self):
        names = judge.parse_criteria_names((ROOT / "rubric.md").read_text())
        self.assertEqual(names, ["壊してはいけない条件", "構成要素の選定", "主要フローの具体化", "異常系の想定", "スケールの議論"])

    def test_raises_when_names_are_incomplete(self):
        with self.assertRaisesRegex(judge.JudgeError, "観点名"):
            judge.parse_criteria_names("## 観点1: 名前（10点）\n")


class CustomIdTest(unittest.TestCase):
    def test_round_trips(self):
        cid = judge.make_custom_id("qwen35-9b-tools", 2, "opus55", 1)
        self.assertEqual(cid, "qwen35-9b-tools__q2__opus55__r1")
        self.assertEqual(judge.split_custom_id(cid), ("qwen35-9b-tools", 2, "opus55", 1))

    def test_rejects_underscore_in_dir(self):
        with self.assertRaisesRegex(judge.JudgeError, "使えない文字"):
            judge.make_custom_id("bad_dir", 1, "opus55", 0)

    def test_accepts_exactly_64_chars(self):
        self.assertEqual(len(judge.make_custom_id("a" * 48, 1, "opus55", 0)), 64)

    def test_rejects_65_chars(self):
        with self.assertRaisesRegex(judge.JudgeError, "64文字"):
            judge.make_custom_id("a" * 49, 1, "opus55", 0)

    def test_split_rejects_other_shapes(self):
        with self.assertRaisesRegex(judge.JudgeError, "形式"):
            judge.split_custom_id("foo")


class BuildRequestTest(unittest.TestCase):
    def test_builds_batch_request_with_cached_rubric_and_schema(self):
        req = judge.build_request("m__q1__opus55__r0", "claude-opus-5-5", "RUBRIC", "出題文", "回答本文")
        self.assertEqual(req["custom_id"], "m__q1__opus55__r0")
        p = req["params"]
        self.assertEqual(p["model"], "claude-opus-5-5")
        self.assertEqual(p["max_tokens"], 4096)
        self.assertEqual(p["system"][0], {"type": "text", "text": "RUBRIC", "cache_control": {"type": "ephemeral"}})
        self.assertEqual(p["system"][1], {"type": "text", "text": judge.RULES})
        self.assertEqual(p["messages"], [{"role": "user", "content": "出題:\n出題文\n\n<answer>\n回答本文\n</answer>"}])
        self.assertEqual(p["output_config"], {"format": {"type": "json_schema", "schema": judge.SCHEMA}})

    def test_schema_requires_all_fields(self):
        self.assertEqual(judge.SCHEMA["required"], ["truncated", "criteria"])
        item = judge.SCHEMA["properties"]["criteria"]["items"]
        self.assertEqual(item["required"], ["id", "score", "evidence", "reason"])
        self.assertFalse(item["additionalProperties"])


def make_judgment(scores, evidence=None, truncated=False):
    return {
        "truncated": truncated,
        "criteria": [
            {"id": i, "score": s, "evidence": list(evidence or []), "reason": "r"}
            for i, s in enumerate(scores, start=1)
        ],
    }


class ValidateJudgmentTest(unittest.TestCase):
    ANSWER = "冒頭で二重販売を防ぐ。\n座席は  Redisで\n確保する。"

    def check(self, obj):
        return judge.validate_judgment(json.dumps(obj, ensure_ascii=False), self.ANSWER)

    def test_accepts_valid_judgment_with_boundary_scores(self):
        obj = make_judgment([0, 10, 5, 5, 5], ["二重販売を防ぐ"])
        self.assertEqual(self.check(obj), (obj, None, []))

    def test_rejects_broken_json(self):
        judgment, error, _ = judge.validate_judgment('{"truncated": false, "crit', self.ANSWER)
        self.assertIsNone(judgment)
        self.assertIn("JSON", error)

    def test_rejects_non_object(self):
        self.assertIn("criteria", self.check([1, 2])[1])

    def test_rejects_duplicate_or_missing_criterion(self):
        obj = make_judgment([1, 2, 3, 4, 5])
        obj["criteria"][4]["id"] = 4
        self.assertIn("重複か欠落", self.check(obj)[1])
        obj["criteria"].pop()
        self.assertIn("重複か欠落", self.check(obj)[1])

    def test_rejects_out_of_range_and_non_integer_scores(self):
        for bad in (11, -1, 5.0, True, "5"):
            obj = make_judgment([1, 2, 3, 4, 5])
            obj["criteria"][2]["score"] = bad
            self.assertIn("0〜10の整数", self.check(obj)[1], bad)

    def test_flags_fabricated_quote(self):
        obj = make_judgment([1, 2, 3, 4, 5], ["存在しない文"])
        judgment, error, fabricated = self.check(obj)
        self.assertIsNone(judgment)
        self.assertIn("引用", error)
        self.assertEqual(fabricated, ["存在しない文"] * 5)

    def test_treats_whitespace_difference_as_match(self):
        obj = make_judgment([1, 2, 3, 4, 5], ["座席は Redisで 確保する。"])
        self.assertIsNone(self.check(obj)[1])

    def test_ignores_blank_quotes(self):
        obj = make_judgment([1, 2, 3, 4, 5], ["", "  "])
        self.assertIsNone(self.check(obj)[1])


if __name__ == "__main__":
    unittest.main()
