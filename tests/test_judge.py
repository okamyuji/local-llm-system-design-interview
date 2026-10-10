import contextlib
import io
import json
import os
import re
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
        assert key == KEY, key
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


NAMES = ["壊してはいけない条件", "構成要素の選定", "主要フローの具体化", "異常系の想定", "スケールの議論"]
TITLES = {1: "題1", 2: "題2", 3: "題3"}


def ok_record(scores, truncated=False):
    return {"status": "ok", "judgment": make_judgment(scores, ["引用"], truncated)}


class RenderScoringTest(unittest.TestCase):
    def test_output_parses_back_with_hand_score_parser(self):
        records = {1: ok_record([1, 2, 3, 4, 5]), 2: ok_record([6, 7, 8, 9, 10]), 3: ok_record([0, 0, 0, 0, 1])}
        text = judge.render_scoring("m", "claude-opus-5-5", "msgbatch_01abc", TITLES, NAMES, records)
        self.assertEqual(judge.parse_hand_scores(text), {1: [1, 2, 3, 4, 5], 2: [6, 7, 8, 9, 10], 3: [0, 0, 0, 0, 1]})
        self.assertIn("claude-opus-5-5", text)
        self.assertIn("msgbatch_01abc", text)
        self.assertIn("- 観点1・壊してはいけない条件: 1点。r「引用」", text)
        self.assertIn("- 合計: 56/150", text)

    def test_notes_truncated_answer(self):
        records = {q: ok_record([1, 1, 1, 1, 1], truncated=(q == 2)) for q in (1, 2, 3)}
        text = judge.render_scoring("m", "claude-opus-5-5", "msgbatch_01abc", TITLES, NAMES, records)
        self.assertEqual(text.count("途中で切れている"), 1)

    def test_marks_missing_question_and_omits_total(self):
        records = {1: ok_record([1, 1, 1, 1, 1]), 2: {"status": "errored"}}
        text = judge.render_scoring("m", "claude-opus-5-5", "msgbatch_01abc", TITLES, NAMES, records)
        self.assertIn("## Q2 題2: 判定なし（errored）", text)
        self.assertIn("## Q3 題3: 判定なし（missing）", text)
        self.assertNotIn("合計", text)


class CliTest(unittest.TestCase):
    DIR = "qwen35-9b"

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        for name in ("rubric.md", "questions.md"):
            shutil.copy(ROOT / name, self.root / name)
        shutil.copytree(ROOT / "results" / self.DIR, self.root / "results" / self.DIR)
        self.env = {"ANTHROPIC_API_KEY": KEY}
        blocker = mock.patch("urllib.request.urlopen", side_effect=AssertionError("テストから実APIへ通信しようとした"))
        blocker.start()
        self.addCleanup(blocker.stop)

    def run_main(self, argv, http=None, env=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = judge.main(argv, root=self.root, http=http or fake_http({}), env=self.env if env is None else env)
        return code, out.getvalue(), err.getvalue()

    def ended_http(self, qs=(1, 2, 3), errored=(), scores=None):
        hand = judge.parse_hand_scores((self.root / "results" / self.DIR / "scoring.md").read_text())
        lines = []
        for q in qs:
            cid = f"{self.DIR}__q{q}__sonnet55__r0"
            if q in errored:
                lines.append(json.dumps({"custom_id": cid, "result": {"type": "errored", "error": {}}}))
                continue
            answer = (self.root / "results" / self.DIR / f"q{q}_raw.txt").read_text()
            judgment = make_judgment((scores or {}).get(q, hand[q]), [answer.strip()[:10]])
            lines.append(json.dumps(succeeded(cid, judgment), ensure_ascii=False))
        return fake_http({
            ("GET", f"{judge.API}/msgbatch_01abc"): b'{"processing_status": "ended", "results_url": "https://r", "request_counts": {}}',
            ("GET", "https://r"): "\n".join(lines).encode(),
        })

    def test_submit_requires_api_key(self):
        code, _, err = self.run_main(["submit"], env={})
        self.assertEqual(code, 1)
        self.assertIn("ANTHROPIC_API_KEY", err)

    def test_submit_builds_one_request_per_answer_model_and_run(self):
        http = fake_http({("POST", judge.API): b'{"id": "msgbatch_01abc"}'})
        code, out, _ = self.run_main(["submit", "--model", "claude-opus-5-5", "--model", "claude-sonnet-5-5", "--runs", "2"], http)
        self.assertEqual(code, 0)
        self.assertIn("msgbatch_01abc", out)
        ids = [r["custom_id"] for r in http.calls[0][2]["requests"]]
        self.assertEqual(len(ids), 12)
        self.assertIn("qwen35-9b__q3__opus55__r1", ids)

    def test_submit_defaults_to_one_run_of_default_model(self):
        http = fake_http({("POST", judge.API): b'{"id": "msgbatch_01abc"}'})
        self.run_main(["submit"], http)
        reqs = http.calls[0][2]["requests"]
        self.assertEqual({r["params"]["model"] for r in reqs}, {judge.DEFAULT_MODEL})
        self.assertEqual(len(reqs), 3)

    def test_submit_rejects_bad_dir_without_sending(self):
        bad = self.root / "results" / "bad_dir"
        shutil.copytree(self.root / "results" / self.DIR, bad)
        http = fake_http({})
        code, _, err = self.run_main(["submit"], http)
        self.assertEqual(code, 1)
        self.assertIn("bad_dir", err)
        self.assertEqual(http.calls, [])

    def test_collect_exits_2_while_processing(self):
        http = fake_http({("GET", f"{judge.API}/msgbatch_01abc"): b'{"processing_status": "in_progress", "request_counts": {"processing": 3}}'})
        code, out, _ = self.run_main(["collect", "msgbatch_01abc"], http)
        self.assertEqual(code, 2)
        self.assertIn("in_progress", out)

    def test_collect_saves_records_and_writes_draft_without_touching_hand_scores(self):
        scoring = self.root / "results" / self.DIR / "scoring.md"
        before = scoring.read_bytes()
        code, out, _ = self.run_main(["collect", "msgbatch_01abc"], self.ended_http())
        self.assertEqual(code, 0)
        self.assertEqual(len(list((self.root / "judge-out" / "msgbatch_01abc").glob("*.json"))), 3)
        draft = (self.root / "results" / self.DIR / "scoring.judge.md").read_text()
        self.assertEqual(judge.parse_hand_scores(draft), judge.parse_hand_scores(before.decode()))
        self.assertEqual(scoring.read_bytes(), before)
        self.assertIn("ok 3", out)

    def test_collect_rejects_path_like_batch_id(self):
        code, _, err = self.run_main(["collect", "../x"])
        self.assertEqual(code, 1)
        self.assertIn("batch ID", err)

    def test_calibrate_passes_when_judge_matches_hand_scores(self):
        self.run_main(["collect", "msgbatch_01abc"], self.ended_http())
        code, out, _ = self.run_main(["calibrate", "msgbatch_01abc"])
        self.assertEqual(code, 0)
        self.assertIn("sonnet55: 合格", out)

    def test_calibrate_reports_undecidable_when_a_judgment_is_missing(self):
        self.run_main(["collect", "msgbatch_01abc"], self.ended_http())
        (self.root / "judge-out" / "msgbatch_01abc" / f"{self.DIR}__q2__sonnet55__r0.json").unlink()
        code, out, _ = self.run_main(["calibrate", "msgbatch_01abc"])
        self.assertEqual(code, 0)
        self.assertIn("sonnet55: 判定不能（判定が欠けた回答 1件）\n", out)

    def test_calibrate_fails_clearly_without_hand_scores(self):
        self.run_main(["collect", "msgbatch_01abc"], self.ended_http())
        (self.root / "results" / self.DIR / "scoring.md").unlink()
        code, _, err = self.run_main(["calibrate", "msgbatch_01abc"])
        self.assertEqual(code, 1)
        self.assertIn("scoring.md", err)

    def test_calibrate_fails_clearly_without_collected_records(self):
        code, _, err = self.run_main(["calibrate", "msgbatch_01zzz"])
        self.assertEqual(code, 1)
        self.assertIn("collect", err)


class EdgeInputTest(unittest.TestCase):
    def test_last_criterion_at_end_of_text_without_newline(self):
        text = "## Q1 題: 15/50\n\n" + "\n".join(f"- 観点{i}・x: {i}点" for i in range(1, 6))
        self.assertEqual(judge.parse_hand_scores(text), {1: [1, 2, 3, 4, 5]})

    def test_parse_error_message_is_exact(self):
        with self.assertRaises(judge.JudgeError) as cm:
            judge.parse_hand_scores("# 空\n")
        self.assertEqual(str(cm.exception), "採点の見出し（## Q<n> ...: <点>/50）がありません")

    def test_normalize_ws_collapses_runs_to_one_space(self):
        self.assertEqual(judge.normalize_ws(" a \n\t b  c "), "a b c")

    def test_schema_messages_are_exact_and_non_dict_criteria_do_not_crash(self):
        self.assertEqual(judge.check_schema([]), "criteriaがありません")
        self.assertEqual(judge.check_schema({"criteria": [1, 2, 3, 4, 5]}),
                         "観点の重複か欠落があります: [None, None, None, None, None]")

    def test_criterion_without_evidence_has_no_fabricated_quotes(self):
        obj = make_judgment([1, 2, 3, 4, 5])
        del obj["criteria"][0]["evidence"]
        self.assertEqual(judge.find_fabricated(obj, "x"), [])

    def test_validation_messages_are_exact(self):
        self.assertEqual(judge.validate_judgment("{", "a")[1], "JSONとして読めません")
        bad = json.dumps(make_judgment([1, 2, 3, 4, 5], ["ない"]))
        self.assertEqual(judge.validate_judgment(bad, "a")[1], "引用が回答本文にありません")

    def test_verdict_reasons_are_exact(self):
        m = {"band_agree": 0.5, "mae": 2.0, "bias": 2.0, "total_within": 0.5}
        self.assertEqual(judge.verdict(m, 0.5, 1), [
            "レンジ一致が80%未満", "平均絶対誤差が1.5点超", "合計差5点以内の回答が83%未満",
            "採点ごとの幅2点以内が90%未満", "引用の捏造が1件", "偏りが±1.5点超"])


class HttpDetailTest(unittest.TestCase):
    def test_get_sends_no_body_with_json_header_and_60s_timeout(self):
        res = mock.MagicMock()
        res.__enter__.return_value.read.return_value = b"{}"
        with mock.patch("urllib.request.urlopen", return_value=res) as urlopen:
            judge.http_request("GET", judge.API, KEY)
        req = urlopen.call_args.args[0]
        self.assertIsNone(req.data)
        self.assertEqual(req.get_method(), "GET")
        self.assertEqual(req.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs, {"timeout": 60})

    def test_undecodable_error_body_is_replaced_not_raised(self):
        err = urllib.error.HTTPError(judge.API, 500, "x", {}, mock.Mock(read=lambda: b"\xff"))
        with mock.patch("urllib.request.urlopen", side_effect=err):
            with self.assertRaisesRegex(judge.JudgeError, "^HTTP 500: �$"):
                judge.http_request("GET", judge.API, KEY)


class ClassifyDetailTest(unittest.TestCase):
    ANSWERS = {("m", 1): "二重販売を防ぐ。"}

    def test_joins_text_blocks_and_skips_others(self):
        payload = json.dumps(make_judgment([1, 2, 3, 4, 5]))
        content = [{"type": "text"}, {"type": "text", "text": payload[:10]}, {"type": "tool_use"},
                   {"type": "text", "text": payload[10:]}]
        result = {"custom_id": "m__q1__opus55__r0", "result": {"type": "succeeded", "message": {"content": content}}}
        self.assertEqual(judge.classify(result, self.ANSWERS)["status"], "ok")

    def test_invalid_record_keeps_error_text(self):
        rec = judge.classify(succeeded("m__q1__opus55__r0", "{"), self.ANSWERS)
        self.assertEqual(rec["error"], "JSONとして読めません")


class RenderLayoutTest(unittest.TestCase):
    def test_exact_layout(self):
        records = {1: {"status": "ok", "judgment": make_judgment([1, 2, 3, 4, 5], ["甲", "乙"], True)},
                   2: {"status": "errored"}}
        text = judge.render_scoring("m", "M", "msgbatch_01abc", TITLES, NAMES, records)
        items = "".join(f"- 観点{i}・{NAMES[i - 1]}: {i}点。r「甲」「乙」\n" for i in range(1, 6))
        self.assertEqual(text, (
            "# m 自動採点の下書き\n\n"
            "判定モデルはM、batch IDはmsgbatch_01abcです。人が回答と照合してから`scoring.md`へ反映してください。\n\n"
            "## Q1 題1: 15/50\n\n回答は途中で切れていると判定しました。\n\n" + items +
            "\n## Q2 題2: 判定なし（errored）\n\n## Q3 題3: 判定なし（missing）\n"))


class CliDetailTest(unittest.TestCase):
    DIR = CliTest.DIR
    setUp = CliTest.setUp
    run_main = CliTest.run_main
    ended_http = CliTest.ended_http

    def run_exit(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err), \
                mock.patch.dict(os.environ, {"COLUMNS": "200"}), self.assertRaises(SystemExit) as cm:
            judge.main(argv, root=self.root, http=fake_http({}), env=self.env)
        return cm.exception.code, out.getvalue(), err.getvalue()

    def test_submit_sends_rubric_question_and_answer_for_each_question(self):
        http = fake_http({("POST", judge.API): b'{"id": "msgbatch_01abc"}'})
        self.run_main(["submit", "--runs", "1"], http)
        reqs = http.calls[0][2]["requests"]
        self.assertEqual([r["custom_id"] for r in reqs], [f"{self.DIR}__q{q}__sonnet55__r0" for q in (1, 2, 3)])
        rubric = (self.root / "rubric.md").read_text()
        questions = judge.parse_questions((self.root / "questions.md").read_text())
        for q, r in zip((1, 2, 3), reqs):
            answer = (self.root / "results" / self.DIR / f"q{q}_raw.txt").read_text()
            self.assertEqual(r["params"]["system"][0]["text"], rubric)
            self.assertEqual(r["params"]["messages"][0]["content"], f"出題:\n{questions[q][1]}\n\n<answer>\n{answer}\n</answer>")

    def test_collect_can_run_twice_and_saves_readable_json(self):
        for _ in range(2):
            code, out, _ = self.run_main(["collect", "msgbatch_01abc"], self.ended_http())
            self.assertEqual(code, 0)
        self.assertIn("judge-out", os.listdir(self.root))
        saved = (self.root / "judge-out" / "msgbatch_01abc" / f"{self.DIR}__q1__sonnet55__r0.json").read_text()
        self.assertEqual(saved, json.dumps(json.loads(saved), ensure_ascii=False, indent=2))
        draft_dir = self.root / "results" / self.DIR
        self.assertIn("scoring.judge.md", os.listdir(draft_dir))
        self.assertIn(f"下書きを書きました: {draft_dir / 'scoring.judge.md'}\n", out)

    def test_collect_reports_mixed_statuses_and_marks_errored_question(self):
        code, out, _ = self.run_main(["collect", "msgbatch_01abc"], self.ended_http(errored=(2,)))
        self.assertIn("errored 1 / ok 2（保存先 ", out)
        draft = (self.root / "results" / self.DIR / "scoring.judge.md").read_text()
        self.assertTrue(draft.startswith(f"# {self.DIR} 自動採点の下書き\n\n判定モデルはclaude-sonnet-5-5、batch IDはmsgbatch_01abcです。"))
        self.assertIn("判定なし（errored）", draft)

    def test_collect_handles_single_result(self):
        code, _, _ = self.run_main(["collect", "msgbatch_01abc"], self.ended_http(qs=(1,)))
        self.assertEqual(code, 0)

    def test_calibrate_reports_every_failed_condition(self):
        self.run_main(["collect", "msgbatch_01abc"], self.ended_http(scores={1: [10, 10, 10, 10, 10]}))
        path = self.root / "judge-out" / "msgbatch_01abc" / f"{self.DIR}__q2__sonnet55__r0.json"
        rec = json.loads(path.read_text())
        path.write_text(json.dumps({**rec, "fabricated": ["x"]}, ensure_ascii=False))
        code, out, _ = self.run_main(["calibrate", "msgbatch_01abc"])
        self.assertEqual(out, (
            "sonnet55: 不合格（レンジ一致が80%未満、平均絶対誤差が1.5点超、合計差5点以内の回答が83%未満、引用の捏造が1件、偏りが±1.5点超）\n"
            "  レンジ一致 66.7% / 平均絶対誤差 1.87 / 偏り +1.87 / 合計差5点以内 66.7% / 1回の幅2点以内 100.0% / 捏造 1件\n"))

    def test_missing_key_message_is_exact(self):
        _, _, err = self.run_main(["collect", "msgbatch_01abc"], env={})
        self.assertEqual(err, "エラー: 環境変数ANTHROPIC_API_KEYを設定してください\n")

    def test_runs_must_be_positive(self):
        code, _, err = self.run_exit(["submit", "--runs", "0"])
        self.assertEqual(code, 2)
        self.assertTrue(err.endswith("argument --runs: 1以上を指定してください\n"), err)

    def test_subcommand_is_required(self):
        self.assertEqual(self.run_exit([])[0], 2)

    def test_unknown_model_is_rejected(self):
        self.assertEqual(self.run_exit(["submit", "--model", "gpt"])[0], 2)

    def test_help_texts(self):
        _, out, _ = self.run_exit(["--help"])
        self.assertIn("\nrubric.mdで回答を採点した下書きを、Message Batches APIで作ります。\n", out)
        for name, text in (("submit", "回答をまとめて1つのバッチに送る"), ("collect", "終わったバッチの結果を保存し、下書きを書く"),
                           ("calibrate", "手採点と比べて合否を出す")):
            self.assertRegex(out, rf"(?m)^\s+{name}\s+{text}$")
        _, out, _ = self.run_exit(["submit", "--help"])
        self.assertIn("{claude-opus-5-5,claude-sonnet-5-5}", out)
        self.assertRegex(out, r"\s判定モデル（既定 claude-sonnet-5-5、複数指定可）\n")
        self.assertRegex(out, r"--runs RUNS\s+同じ回答を採点する回数\n")
        self.assertRegex(out, r"\s採点例にする手採点済みディレクトリ（省略時は対象以外の手採点すべて、複数指定可）\n")
        self.assertRegex(out, r"dirs\s+results/配下の対象ディレクトリ（省略時は全部）\n")


class ExamplesAndMarkupTest(unittest.TestCase):
    def test_markdown_emphasis_backticks_and_outer_brackets_are_not_fabrication(self):
        obj = make_judgment([1, 2, 3, 4, 5], ["「ロードバランシング: NGINXで分散」", "Redisの INCR"])
        self.assertEqual(judge.find_fabricated(obj, "**ロードバランシング**: NGINXで分散する。`Redis`の `INCR`"), [])

    def test_elided_quote_is_still_fabrication(self):
        obj = make_judgment([1, 2, 3, 4, 5], ["ロードバランシング (…) 分散"])
        self.assertEqual(len(judge.find_fabricated(obj, "**ロードバランシング**: NGINXで分散する。")), 5)

    def test_build_request_puts_cached_examples_between_rubric_and_rules(self):
        system = judge.build_request("m__q1__opus55__r0", "claude-opus-5-5", "RUBRIC", "Q", "A", "EX")["params"]["system"]
        self.assertEqual(system, [
            {"type": "text", "text": "RUBRIC", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "EX", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": judge.RULES},
        ])

    def test_load_examples_wraps_each_hand_scoring(self):
        text = judge.load_examples(ROOT, ["qwen35-9b"])
        scoring = (ROOT / "results" / "qwen35-9b" / "scoring.md").read_text()
        self.assertTrue(text.startswith("次は手採点の採点例です。採点の水準をこの例に合わせてください。\n\n"))
        self.assertIn(f'<example dir="qwen35-9b">\n{scoring}\n</example>', text)
        self.assertEqual(judge.load_examples(ROOT, []), "")

    def test_load_examples_separates_examples_with_blank_line(self):
        text = judge.load_examples(ROOT, ["qwen35-9b", "gemma4-e4b"])
        self.assertIn("\n</example>\n\n<example dir=\"gemma4-e4b\">\n", text)

    def test_only_outer_corner_brackets_are_stripped(self):
        obj = make_judgment([1, 2, 3, 4, 5], ["X社"])
        self.assertEqual(judge.find_fabricated(obj, "社のX"), ["X社"] * 5)

    def test_load_examples_requires_hand_scoring(self):
        with self.assertRaisesRegex(judge.JudgeError, "手採点がありません"):
            judge.load_examples(ROOT, ["no-such-dir"])


class SubmitExamplesTest(unittest.TestCase):
    DIR = CliTest.DIR
    setUp = CliTest.setUp
    run_main = CliTest.run_main

    def add_other_dir(self):
        shutil.copytree(self.root / "results" / self.DIR, self.root / "results" / "other-model")

    def submit(self, argv):
        http = fake_http({("POST", judge.API): b'{"id": "msgbatch_01abc"}'})
        code, _, err = self.run_main(argv, http)
        return code, err, http

    def test_default_examples_are_hand_scored_dirs_other_than_targets(self):
        self.add_other_dir()
        code, _, http = self.submit(["submit", self.DIR])
        self.assertEqual(code, 0)
        examples = http.calls[0][2]["requests"][0]["params"]["system"][1]["text"]
        self.assertIn('<example dir="other-model">', examples)
        self.assertNotIn(f'<example dir="{self.DIR}">', examples)

    def test_no_examples_block_when_nothing_else_is_hand_scored(self):
        code, _, http = self.submit(["submit", self.DIR])
        self.assertEqual(len(http.calls[0][2]["requests"][0]["params"]["system"]), 2)

    def test_explicit_examples_overlapping_targets_are_rejected_without_sending(self):
        code, err, http = self.submit(["submit", "--examples", self.DIR, self.DIR])
        self.assertEqual(code, 1)
        self.assertEqual(err, f"エラー: 採点対象を採点例に含めることはできません: ['{self.DIR}']\n")
        self.assertEqual(http.calls, [])

    def test_explicit_examples_are_used(self):
        self.add_other_dir()
        code, _, http = self.submit(["submit", "--examples", "other-model", self.DIR])
        self.assertEqual(code, 0)
        self.assertIn('<example dir="other-model">', http.calls[0][2]["requests"][0]["params"]["system"][1]["text"])


if __name__ == "__main__":
    unittest.main()
