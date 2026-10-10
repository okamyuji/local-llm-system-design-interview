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


PASSING = {"band_agree": 0.8, "mae": 1.5, "bias": 1.5, "total_within": 20 / 24}


class MetricsTest(unittest.TestCase):
    def test_band_boundaries(self):
        self.assertEqual([judge.band(s) for s in (0, 3, 4, 7, 8, 10)], [0, 0, 1, 1, 2, 2])

    def test_compare_on_hand_computed_example(self):
        m = judge.compare({("m", 1): [1, 4, 7, 8, 10]}, {("m", 1): [3, 4, 8, 8, 6]})
        self.assertAlmostEqual(m["band_agree"], 0.6)
        self.assertAlmostEqual(m["mae"], 1.4)
        self.assertAlmostEqual(m["bias"], -0.2)
        self.assertEqual(m["total_within"], 1.0)

    def test_total_within_boundary_is_five_points(self):
        hand = {("m", 1): [5, 5, 5, 5, 5], ("m", 2): [5, 5, 5, 5, 5]}
        judged = {("m", 1): [10, 5, 5, 5, 5], ("m", 2): [10, 6, 5, 5, 5]}
        self.assertEqual(judge.compare(hand, judged)["total_within"], 0.5)

    def test_stability_counts_spans_up_to_two(self):
        runs = [{("m", 1): [5, 5, 0, 0, 0]}, {("m", 1): [7, 8, 0, 0, 0]}]
        self.assertAlmostEqual(judge.stability(runs), 0.8)

    def test_verdict_passes_exactly_at_thresholds(self):
        self.assertEqual(judge.verdict(PASSING, 0.9, 0), [])
        self.assertEqual(judge.verdict({**PASSING, "bias": -1.5}, 0.9, 0), [])

    def test_verdict_fails_just_past_each_threshold(self):
        cases = [
            ({**PASSING, "band_agree": 0.79}, 0.9, 0, "レンジ一致"),
            ({**PASSING, "mae": 1.51}, 0.9, 0, "平均絶対誤差"),
            ({**PASSING, "total_within": 19 / 24}, 0.9, 0, "合計差"),
            (PASSING, 0.89, 0, "幅"),
            (PASSING, 0.9, 1, "捏造"),
            ({**PASSING, "bias": 1.51}, 0.9, 0, "偏り"),
            ({**PASSING, "bias": -1.51}, 0.9, 0, "偏り"),
        ]
        for m, stable, fab, word in cases:
            reasons = judge.verdict(m, stable, fab)
            self.assertEqual(len(reasons), 1, word)
            self.assertIn(word, reasons[0])


KEY = "not-a-real-key"


def fake_http(responses):
    calls = []

    def http(method, url, key, body=None):
        calls.append((method, url, body))
        return responses[(method, url)]

    http.calls = calls
    return http


def succeeded(cid, payload):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"custom_id": cid, "result": {"type": "succeeded", "message": {"content": [{"type": "text", "text": text}]}}}


class HttpRequestTest(unittest.TestCase):
    def test_sends_method_headers_and_json_body(self):
        res = mock.MagicMock()
        res.__enter__.return_value.read.return_value = b"{}"
        with mock.patch("urllib.request.urlopen", return_value=res) as urlopen:
            self.assertEqual(judge.http_request("POST", judge.API, KEY, {"a": 1}), b"{}")
        req = urlopen.call_args.args[0]
        self.assertEqual(req.get_method(), "POST")
        self.assertEqual(req.get_header("X-api-key"), KEY)
        self.assertEqual(req.get_header("Anthropic-version"), "2023-06-01")
        self.assertEqual(json.loads(req.data), {"a": 1})

    def test_wraps_http_error_with_status_and_body(self):
        err = urllib.error.HTTPError(judge.API, 401, "Unauthorized", {}, mock.Mock(read=lambda: b"invalid x-api-key"))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaisesRegex(judge.JudgeError, "HTTP 401: invalid x-api-key"):
                judge.http_request("GET", judge.API, KEY)

    def test_wraps_connection_error(self):
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("no route")):
            with self.assertRaisesRegex(judge.JudgeError, "接続できません"):
                judge.http_request("GET", judge.API, KEY)


class BatchOpsTest(unittest.TestCase):
    def test_create_batch_posts_requests_and_returns_id(self):
        http = fake_http({("POST", judge.API): b'{"id": "msgbatch_01abc"}'})
        self.assertEqual(judge.create_batch([{"custom_id": "x"}], KEY, http), "msgbatch_01abc")
        self.assertEqual(http.calls, [("POST", judge.API, {"requests": [{"custom_id": "x"}]})])

    def test_get_batch_rejects_path_like_id_without_calling_api(self):
        http = fake_http({})
        for bad in ("../x", "msgbatch_01/../a", ""):
            with self.assertRaisesRegex(judge.JudgeError, "batch ID"):
                judge.get_batch(bad, KEY, http)
        self.assertEqual(http.calls, [])

    def test_get_results_parses_jsonl_and_skips_blank_lines(self):
        http = fake_http({("GET", "https://r"): b'{"a": 1}\n\n{"b": 2}\n'})
        self.assertEqual(judge.get_results("https://r", KEY, http), [{"a": 1}, {"b": 2}])


class ClassifyTest(unittest.TestCase):
    ANSWERS = {("m", 1): "二重販売を防ぐ。"}

    def test_ok(self):
        rec = judge.classify(succeeded("m__q1__opus55__r0", make_judgment([1, 2, 3, 4, 5], ["二重販売"])), self.ANSWERS)
        self.assertEqual((rec["status"], rec["dir"], rec["q"], rec["model"], rec["run"]), ("ok", "m", 1, "opus55", 0))
        self.assertEqual(rec["judgment"]["criteria"][0]["score"], 1)

    def test_errored_is_kept_without_judgment(self):
        rec = judge.classify({"custom_id": "m__q1__opus55__r0", "result": {"type": "errored", "error": {}}}, self.ANSWERS)
        self.assertEqual(rec["status"], "errored")
        self.assertNotIn("judgment", rec)

    def test_fabricated_is_invalid(self):
        rec = judge.classify(succeeded("m__q1__opus55__r0", make_judgment([1, 2, 3, 4, 5], ["ない"])), self.ANSWERS)
        self.assertEqual(rec["status"], "invalid")
        self.assertEqual(len(rec["fabricated"]), 5)

    def test_cut_off_output_is_invalid_not_crash(self):
        rec = judge.classify(succeeded("m__q1__opus55__r0", '{"truncated": fal'), self.ANSWERS)
        self.assertEqual(rec["status"], "invalid")
        self.assertEqual(rec["raw"], '{"truncated": fal')


if __name__ == "__main__":
    unittest.main()
