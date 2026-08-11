import datetime as dt
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import eodhd_client  # noqa: E402


class ApiKeyTests(unittest.TestCase):
    def test_raises_without_any_key(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(eodhd_client.EODHDError):
                eodhd_client._api_key()

    def test_falls_back_to_eodhd_api_key(self):
        with mock.patch.dict("os.environ", {"EODHD_API_KEY": "abc"}, clear=True):
            self.assertEqual(eodhd_client._api_key(), "abc")

    def test_falls_back_to_eodhd_api_token(self):
        with mock.patch.dict("os.environ", {"EODHD_API_TOKEN": "xyz"}, clear=True):
            self.assertEqual(eodhd_client._api_key(), "xyz")

    def test_custom_env_var_takes_priority(self):
        with mock.patch.dict("os.environ", {"EODHD_API_KEY": "generic", "MY_STRATEGY_KEY": "specific"}, clear=True):
            self.assertEqual(eodhd_client._api_key(env_var="MY_STRATEGY_KEY"), "specific")

    def test_custom_env_var_falls_through_if_unset(self):
        with mock.patch.dict("os.environ", {"EODHD_API_KEY": "generic"}, clear=True):
            self.assertEqual(eodhd_client._api_key(env_var="MY_STRATEGY_KEY"), "generic")


class ChunkRangesTests(unittest.TestCase):
    def test_single_chunk_when_within_limit(self):
        chunks = eodhd_client._chunk_ranges(dt.date(2024, 1, 1), dt.date(2024, 1, 10), max_days=120)
        self.assertEqual(chunks, [(dt.date(2024, 1, 1), dt.date(2024, 1, 10))])

    def test_splits_at_120_day_boundary(self):
        start = dt.date(2024, 1, 1)
        end = start + dt.timedelta(days=250)
        chunks = eodhd_client._chunk_ranges(start, end, max_days=120)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0][0], start)
        self.assertEqual(chunks[-1][1], end)
        # Inga luckor eller överlapp mellan chunkarna.
        for (s1, e1), (s2, e2) in zip(chunks, chunks[1:]):
            self.assertEqual(s2, e1 + dt.timedelta(days=1))

    def test_chunk_never_exceeds_max_days(self):
        chunks = eodhd_client._chunk_ranges(dt.date(2024, 1, 1), dt.date(2024, 12, 31), max_days=120)
        for s, e in chunks:
            self.assertLessEqual((e - s).days + 1, 120)


class GetEodCachingTests(unittest.TestCase):
    def test_caches_to_disk_and_reuses_without_a_second_http_call(self):
        canned = [
            {"date": "2024-01-02", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05,
             "adjusted_close": 1.05, "volume": 1000},
            {"date": "2024-01-03", "open": 1.05, "high": 1.2, "low": 1.0, "close": 1.15,
             "adjusted_close": 1.15, "volume": 1200},
        ]
        fake_response = mock.Mock()
        fake_response.json.return_value = canned

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"EODHD_API_KEY": "dummy"}, clear=True), \
                mock.patch.object(eodhd_client, "_get", return_value=fake_response) as mocked_get:
            df1 = eodhd_client.get_eod("SPY", start="2024-01-01", end="2024-01-03", cache_dir=tmp)
            df2 = eodhd_client.get_eod("SPY", start="2024-01-01", end="2024-01-03", cache_dir=tmp)

            self.assertEqual(mocked_get.call_count, 1)  # andra anropet togs från disk-cachen
            self.assertEqual(len(df1), 2)
            self.assertListEqual(list(df1.columns), ["open", "high", "low", "close", "adjusted_close", "volume"])
            self.assertTrue(df1.index.equals(df2.index))

            cache_files = list(Path(tmp).glob("eod_*.json"))
            self.assertEqual(len(cache_files), 1)
            with open(cache_files[0]) as f:
                self.assertEqual(json.load(f), canned)

    def test_empty_payload_returns_empty_dataframe_with_expected_columns(self):
        fake_response = mock.Mock()
        fake_response.json.return_value = []
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict("os.environ", {"EODHD_API_KEY": "dummy"}, clear=True), \
                mock.patch.object(eodhd_client, "_get", return_value=fake_response):
            df = eodhd_client.get_eod("NOPE", cache_dir=tmp)
            self.assertEqual(len(df), 0)
            self.assertListEqual(list(df.columns), ["open", "high", "low", "close", "adjusted_close", "volume"])


if __name__ == "__main__":
    unittest.main()
