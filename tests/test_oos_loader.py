import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.loader_guard import LoaderBypassError, loader_context  # noqa: E402
from research.oos_loader import (  # noqa: E402
    OOSLockError,
    _default_synthetic_fetch,
    enforce_oos_gate,
    load_market_data,
)

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
            _default_synthetic_fetch(BASE_CONFIG, "a", unlock_oos=True, log_path=Path("/dev/null"))

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

    def test_injected_fetch_fn_cannot_smuggle_a_different_oos_config(self):
        """Regression: en tidigare version litade bara på att den YTTRE
        configen kontrollerades, vilket lät ett inbytt fetch_fn hämta en
        annan, okontrollerad (och OOS) config obehindrat. enforce_oos_gate
        måste anropas av den faktiska hämtningen på DESS EGEN config för att
        stänga detta."""
        in_sample_config = dict(BASE_CONFIG, data_end="2020-01-10")  # trivialt godkänd av yttre kontroll

        def evil_fetch(config, twin, *, unlock_oos, log_path):
            smuggled = dict(config, data_end="2099-01-01")  # långt bortom is_end
            return _default_synthetic_fetch(smuggled, twin, unlock_oos=False, log_path=log_path)

        with self.assertRaises(OOSLockError):
            load_market_data(in_sample_config, twin="a", unlock_oos=False, fetch_fn=evil_fetch)

    def test_flipping_the_raw_contextvar_does_not_leak_oos_data_silently(self):
        """Regression: att direkt sätta loader_guard-flaggan (ett känt,
        dokumenterat, icke-vattentätt kringgående av kod-konventionen) får
        INTE ensamt räcka för att läsa OOS-data — enforce_oos_gate måste
        fortfarande höja fel om unlock_oos inte anges explicit."""
        oos_config = dict(BASE_CONFIG, data_end="2099-01-01")
        with loader_context():
            with self.assertRaises(OOSLockError):
                _default_synthetic_fetch(oos_config, "a", unlock_oos=False, log_path=Path("/dev/null"))

    def test_enforce_oos_gate_logs_exactly_once_per_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "oos_unlocks.jsonl"
            oos_config = dict(BASE_CONFIG, data_end="2020-06-01")
            load_market_data(oos_config, twin="a", unlock_oos=True, log_path=log_path)
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)

    def test_enforce_oos_gate_blocks_and_logs_independently_of_loader_context(self):
        """enforce_oos_gate() gör själva OOS-kontrollen + loggningen och kräver
        inte loader-kontext (den anropas AV fetch-implementationer, som
        själva kräver kontext via require_loader_active — se
        test_internal_fetch_cannot_be_called_directly)."""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "oos_unlocks.jsonl"
            oos_config = dict(BASE_CONFIG, data_end="2099-01-01")
            with self.assertRaises(OOSLockError):
                enforce_oos_gate(oos_config, False, log_path=log_path)
            self.assertFalse(log_path.exists())
            enforce_oos_gate(oos_config, True, log_path=log_path)
            self.assertEqual(len(log_path.read_text(encoding="utf-8").strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
