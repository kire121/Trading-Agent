import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import registry  # noqa: E402
from lib.hashutil import compute_config_hash  # noqa: E402


def _valid_entry(**overrides):
    entry = {
        "yta_id": "Y_test_panel",
        "tickerlista_sha256": compute_config_hash(sorted(["AAA", "BBB", "CCC"])),
        "tickers": ["AAA", "BBB", "CCC"],
        "period": "2020-01-01..2020-12-31",
        "frekvens": "daglig",
        "kolumner": ["close", "volume"],
        "läsningstyp": "IS",
        "strategi": "test_strategi",
        "datum": "2026-08-11",
        "repo_panel": "test panel",
    }
    entry.update(overrides)
    return entry


class ValidateEntryTests(unittest.TestCase):
    def test_valid_entry_passes(self):
        registry.validate_entry(_valid_entry())  # no raise

    def test_missing_field_raises(self):
        entry = _valid_entry()
        del entry["yta_id"]
        with self.assertRaises(ValueError):
            registry.validate_entry(entry)

    def test_unknown_field_raises(self):
        entry = _valid_entry(extra_field="not in schema")
        with self.assertRaises(ValueError):
            registry.validate_entry(entry)

    def test_wrong_type_raises(self):
        entry = _valid_entry(tickers="AAA,BBB,CCC")  # str, not list
        with self.assertRaises(TypeError):
            registry.validate_entry(entry)

    def test_empty_tickers_raises(self):
        entry = _valid_entry(tickers=[], tickerlista_sha256=compute_config_hash([]))
        with self.assertRaises(ValueError):
            registry.validate_entry(entry)

    def test_ticker_hash_mismatch_raises(self):
        entry = _valid_entry(tickerlista_sha256="0" * 64)
        with self.assertRaises(ValueError):
            registry.validate_entry(entry)

    def test_hash_is_order_independent_sorted_convention(self):
        # tickers not pre-sorted in the entry itself -- the hash is defined
        # over sorted(tickers), so this must still validate.
        entry = _valid_entry(tickers=["CCC", "AAA", "BBB"])
        registry.validate_entry(entry)  # no raise


class AppendAndReadEntriesTests(unittest.TestCase):
    def test_append_then_read_round_trips(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ytor.jsonl"
            entry_a = _valid_entry(yta_id="Y_a")
            entry_b = _valid_entry(yta_id="Y_b")
            registry.append_entry(entry_a, path=path)
            registry.append_entry(entry_b, path=path)

            entries = registry.read_entries(path=path)
            self.assertEqual([e["yta_id"] for e in entries], ["Y_a", "Y_b"])

    def test_append_is_pure_append_never_rewrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ytor.jsonl"
            registry.append_entry(_valid_entry(yta_id="Y_a"), path=path)
            before = path.read_text(encoding="utf-8")
            registry.append_entry(_valid_entry(yta_id="Y_b"), path=path)
            after = path.read_text(encoding="utf-8")
            self.assertTrue(after.startswith(before))

    def test_append_rejects_invalid_entry_without_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ytor.jsonl"
            bad_entry = _valid_entry()
            del bad_entry["strategi"]
            with self.assertRaises(ValueError):
                registry.append_entry(bad_entry, path=path)
            self.assertFalse(path.exists())

    def test_read_entries_missing_file_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "does_not_exist.jsonl"
            self.assertEqual(registry.read_entries(path=path), [])

    def test_each_line_is_valid_standalone_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ytor.jsonl"
            registry.append_entry(_valid_entry(), path=path)
            lines = path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            json.loads(lines[0])  # no raise


class CommittedRegistryFileTests(unittest.TestCase):
    """Regressionsskydd: den faktiska committade registry/ytor.jsonl (Y1,
    Timglasets US-40-ETF-läsning) ska alltid klara samma schemavalidering
    som nya poster måste."""

    def test_committed_ytor_jsonl_entries_all_validate(self):
        entries = registry.read_entries()
        self.assertGreaterEqual(len(entries), 1)
        for entry in entries:
            registry.validate_entry(entry)


if __name__ == "__main__":
    unittest.main()
