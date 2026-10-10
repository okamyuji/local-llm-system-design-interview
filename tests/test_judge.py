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


if __name__ == "__main__":
    unittest.main()
