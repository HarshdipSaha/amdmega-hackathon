from silentpath.mcp_server import summarise_store, which_path
from silentpath.record import Confidence, CostSample, ObservedPath, Output, RunRecord
from silentpath.store import RecordStore


def _record(backend, observed, workload="w", text="same"):
    return RunRecord.build(
        config={"backend": backend}, workload_id=workload,
        path=ObservedPath(requested=backend, observed=observed,
                          confidence=Confidence.REPORTED,
                          signals={"profiler": observed} if observed else {}),
        cost=CostSample(wall_seconds=1, device_seconds=1, samples=3, wall_stdev=.01),
        output=Output(text=text, token_ids=[1], chosen_logprobs=[-.5], decision=text),
        artifacts={"log": "abc.log.txt"},
    )


def test_which_path_reads_jsonl_and_separates_unresolved(tmp_path):
    store = RecordStore(tmp_path)
    store.append(_record("FLASH_ATTENTION", "MATH"))
    store.append(_record("MATH", None, workload="w2"))
    result = which_path(str(tmp_path))
    assert result["records"] == 2
    assert len(result["silent_fallbacks"]) == 1
    assert len(result["unresolved"]) == 1
    assert result["paths"][0]["artifacts"] == {"log": "abc.log.txt"}


def test_summarise_store_counts_comparisons_and_findings(tmp_path):
    store = RecordStore(tmp_path)
    store.append(_record("FLASH_ATTENTION", "FLASH_ATTENTION", text="same"))
    store.append(_record("MATH", "MATH", text="different"))
    result = summarise_store(str(tmp_path))
    assert result["comparisons"] == 1
    assert result["findings"] == 1


def test_empty_store_has_zero_summary(tmp_path):
    assert summarise_store(str(tmp_path))["records"] == 0
