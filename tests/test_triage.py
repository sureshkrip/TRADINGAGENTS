"""Stage 1 triage: batching, JSON parsing robustness, ranking, and safety.

A FakeLLM stands in for the chat model, so no network or tokens are used. The
real LLM plumbing is exercised only indirectly (create_llm_client) and is not
touched here.
"""

import pytest

import tradingagents.funnel.triage as tr
from tradingagents.dataflows.screen import ScreenResult
from tradingagents.funnel.triage import triage_candidates


class _FakeMsg:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Records the batches it was asked to score; replies with a scripted body."""

    def __init__(self, reply_for):
        self.reply_for = reply_for  # callable(batch) -> str
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        # The human message lists the tickers; hand the whole thing to the script.
        return _FakeMsg(self.reply_for(messages))


def _cand(ticker, score=80.0, ret_3m=0.2):
    return ScreenResult(
        ticker=ticker,
        score=score,
        components={},
        metrics={"ret_3m": ret_3m, "ret_6m": 0.3, "above_200sma": 0.1,
                 "trend_struct": 1.0, "dollar_vol": 2e7},
    )


@pytest.mark.unit
def test_empty_input_returns_empty():
    assert triage_candidates([]) == []


@pytest.mark.unit
def test_ranks_by_triage_score_and_truncates():
    cands = [_cand("AAA"), _cand("BBB"), _cand("CCC")]
    reply = '[{"ticker":"AAA","score":90,"thesis":"strong","red_flag":null},' \
            '{"ticker":"BBB","score":40,"thesis":"meh","red_flag":"valuation"},' \
            '{"ticker":"CCC","score":70,"thesis":"ok","red_flag":null}]'
    llm = FakeLLM(lambda m: reply)
    out = triage_candidates(cands, top_n=2, llm=llm)
    assert [r.ticker for r in out] == ["AAA", "CCC"]
    assert out[0].triage_score == 90 and out[0].red_flag is None
    assert out[1].thesis == "ok"


@pytest.mark.unit
def test_parses_fenced_json_and_prose_wrapper():
    cands = [_cand("AAA")]
    reply = "Sure! Here you go:\n```json\n[{\"ticker\":\"AAA\",\"score\":55,\"thesis\":\"t\",\"red_flag\":\"r\"}]\n```\nDone."
    out = triage_candidates(cands, llm=FakeLLM(lambda m: reply))
    assert out[0].triage_score == 55 and out[0].red_flag == "r"


@pytest.mark.unit
def test_drops_hallucinated_tickers():
    cands = [_cand("AAA")]
    reply = '[{"ticker":"AAA","score":60,"thesis":"real","red_flag":null},' \
            '{"ticker":"ZZZ","score":99,"thesis":"made up","red_flag":null}]'
    out = triage_candidates(cands, llm=FakeLLM(lambda m: reply))
    assert [r.ticker for r in out] == ["AAA"]


@pytest.mark.unit
def test_score_clamped_and_carries_screen_score():
    cands = [_cand("AAA", score=77.0)]
    reply = '[{"ticker":"AAA","score":150,"thesis":"t","red_flag":null}]'
    out = triage_candidates(cands, llm=FakeLLM(lambda m: reply))
    assert out[0].triage_score == 100.0        # clamped to [0,100]
    assert out[0].screen_score == 77.0         # Stage 0 score preserved


@pytest.mark.unit
def test_malformed_reply_yields_no_results_not_crash():
    cands = [_cand("AAA")]
    out = triage_candidates(cands, llm=FakeLLM(lambda m: "no json here, sorry"))
    assert out == []


@pytest.mark.unit
def test_batches_large_candidate_lists(monkeypatch):
    monkeypatch.setattr(tr, "_BATCH_SIZE", 2)
    cands = [_cand(f"T{i}") for i in range(5)]  # 5 -> 3 batches of size 2,2,1

    def reply(messages):
        # Echo a score for whatever tickers appear in the human message.
        text = messages[-1].content
        items = [f'{{"ticker":"{c.ticker}","score":50,"thesis":"t","red_flag":null}}'
                 for c in cands if c.ticker in text]
        return "[" + ",".join(items) + "]"

    llm = FakeLLM(reply)
    out = triage_candidates(cands, top_n=10, llm=llm)
    assert llm.calls == 3
    assert len(out) == 5


@pytest.mark.unit
def test_bad_top_n_env_raises(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_TRIAGE_TOP_N", "lots")
    with pytest.raises(ValueError, match="must be an integer"):
        triage_candidates([_cand("AAA")], llm=FakeLLM(lambda m: "[]"))
