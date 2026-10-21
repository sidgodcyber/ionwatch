"""One-minute record schema: batch minutes.jsonl validates; malformed records are rejected."""
import copy
import json

import jsonschema
import pytest
from conftest import GOOD

from ionwatch import pipeline, schema


@pytest.fixture(scope="module")
def batch_records(tmp_path_factory, cfg):
    out = tmp_path_factory.mktemp("batch") / "2026-10-18"
    res = pipeline.run([str(GOOD)], None, out, cfg, plots=False, node_id="test-node")
    return res, schema.read_jsonl(out / "minutes.jsonl")


def test_batch_minutes_jsonl_valid(batch_records):
    res, recs = batch_records
    assert len(recs) == res["minutes"] >= 3
    for r in recs:
        schema.validate(r)
        assert r["node_id"] == "test-node" and r["receiver"] == "phone"
    minutes = [r["minute_utc"] for r in recs]
    assert minutes == sorted(minutes) and len(set(minutes)) == len(minutes)


def test_batch_record_content(batch_records):
    _, recs = batch_records
    full = [r for r in recs if not any(w.startswith("receiver_partial") for w in r["warnings"])]
    assert full
    r = full[0]
    assert r["navic_status"] == "tracked"
    assert r["sats_tracked"] == len({(s["sys"], s["svid"]) for s in r["satellites"] if s["cn0_mean"] is not None})
    bands = {(s["sys"], s["band"]) for s in r["satellites"]}
    assert {("GPS", "L1"), ("GPS", "L5"), ("NAVIC", "L5")} <= bands
    assert any(isinstance(x["agc_flag"], bool) for x in full)     # null during the 60 s AGC warm-up
    assert all(s["s4_flag"] != "valid" or s["s4"] > 2 * s["s4_noise"] for s in r["satellites"])


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("node_id"),
    lambda r: r.update(minute_utc="2026-10-18 08:40"),
    lambda r: r.update(navic_status="maybe"),
    lambda r: r["satellites"][0].update(s4_flag="good"),
    lambda r: r["satellites"][0].update(sys="GALILEO"),
    lambda r: r["satellites"][0].pop("class"),
    lambda r: r.update(extra_field=1),
])
def test_invalid_records_rejected(batch_records, mutate):
    _, recs = batch_records
    r = copy.deepcopy(next(x for x in recs if x["satellites"]))
    mutate(r)
    with pytest.raises(jsonschema.ValidationError):
        schema.validate(r)


def test_schema_file_is_valid_json_schema():
    jsonschema.Draft202012Validator.check_schema(json.loads(schema.SCHEMA_PATH.read_text(encoding="utf-8")))
