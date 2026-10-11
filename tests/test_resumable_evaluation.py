"""Checkpoint helper tests ensure interrupted stages can resume safely."""

from scripts.evaluate_resumable import _atomic_json, _json_checkpoint


def test_resume_reuses_only_a_completed_checkpoint(tmp_path):
    checkpoint = tmp_path / "fold.json"
    _atomic_json(checkpoint, {"fold": 1, "metrics": {"recall": 0.5}})

    def should_not_run():
        raise AssertionError("a valid completed checkpoint should be reused")

    assert _json_checkpoint(checkpoint, True, should_not_run) == {
        "fold": 1, "metrics": {"recall": 0.5}
    }


def test_new_checkpoint_is_written_after_calculation(tmp_path):
    checkpoint = tmp_path / "fold.json"
    result = _json_checkpoint(checkpoint, False, lambda: {"fold": 2})
    assert result == {"fold": 2}
    assert checkpoint.read_text(encoding="utf-8").strip() == '{\n  "fold": 2\n}'
    assert not checkpoint.with_suffix(".json.tmp").exists()
