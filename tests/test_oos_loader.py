import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.loader_guard import LoaderBypassError  # noqa: E402
from research.oos_loader import OOSLockError, _default_synthetic_fetch, load_market_data  # noqa: E402

BASE_CONFIG = {
    "strategy_name": "test_strategy",
    "instrument": "TEST",
    "seed": 1,
    "data_start": "2020-01-01",
    "data_end": "2020-02-01",
    "is_end": "2020-01-15",
}


class OOSLoaderTests(unittest.TestCase):
    def test_lock_blocks_dates_after_is_end(self):
        with self.assertRaises(OOSLockError):
            load_market_data(BASE_CONFIG, twin="a", unlock_oos=False)

    def test_unlock_allows_and_logs(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "oos_unlocks.jsonl"
            bars = load_market_data(BASE_CONFIG, twin="a", unlock_oos=True, log_path=log_path)
            self.assertTrue(len(bars) > 0)
            self.assertTrue(log_path.exists())
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            entry = json.loads(lines[0])
            self.assertEqual(entry["strategy_name"], "test_strategy")
            self.assertIn("config_hash", entry)
            self.assertIn("timestamp_utc", entry)

    def test_in_sample_requires_no_unlock_and_does_not_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "oos_unlocks.jsonl"
            in_sample_config = dict(BASE_CONFIG, data_end="2020-01-10")
            bars = load_market_data(in_sample_config, twin="a", unlock_oos=False, log_path=log_path)
            self.assertTrue(len(bars) > 0)
            self.assertFalse(log_path.exists())

    def test_internal_fetch_cannot_be_called_directly(self):
        with self.assertRaises(LoaderBypassError):
            _default_synthetic_fetch(BASE_CONFIG, "a")

    def test_deterministic_same_seed_same_bars(self):
        in_sample_config = dict(BASE_CONFIG, data_end="2020-01-10")
        bars1 = load_market_data(in_sample_config, twin="a", unlock_oos=False)
        bars2 = load_market_data(in_sample_config, twin="a", unlock_oos=False)
        self.assertEqual(bars1, bars2)

    def test_different_twins_differ(self):
        in_sample_config = dict(BASE_CONFIG, data_end="2020-01-10")
        bars_a = load_market_data(in_sample_config, twin="a", unlock_oos=False)
        bars_b = load_market_data(in_sample_config, twin="b", unlock_oos=False)
        self.assertNotEqual(bars_a, bars_b)


if __name__ == "__main__":
    unittest.main()
