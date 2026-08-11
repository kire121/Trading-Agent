import datetime
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.configvalidate import ConfigError, validate_and_normalize  # noqa: E402

VALID = {
    "strategy_name": "my_strategy",
    "seed": 42,
    "data_start": "2020-01-01",
    "data_end": "2020-06-30",
    "is_end": "2020-12-31",
    "fast_exit_steps": [1, 3, 5],
    "twins": ["twin_a", "twin_b"],
}


class ConfigValidateTests(unittest.TestCase):
    def test_valid_config_passes_unchanged(self):
        result = validate_and_normalize(VALID)
        self.assertEqual(result["strategy_name"], "my_strategy")
        self.assertEqual(result["data_start"], "2020-01-01")

    def test_missing_required_key_raises(self):
        broken = dict(VALID)
        del broken["seed"]
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_yaml_date_object_is_normalized_to_string(self):
        """PyYAML gör ett ociterat YYYY-MM-DD till ett datumobjekt. Detta får
        inte krascha hashning/JSON-skrivning längre fram i pipelinen."""
        with_date_objects = dict(VALID, data_start=datetime.date(2020, 1, 1))
        result = validate_and_normalize(with_date_objects)
        self.assertEqual(result["data_start"], "2020-01-01")
        self.assertIsInstance(result["data_start"], str)

    def test_invalid_date_string_raises(self):
        broken = dict(VALID, data_start="not-a-date")
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_zero_fast_exit_step_raises(self):
        """Regression: steg=0 orsakade tidigare en oändlig loop i simuleringen."""
        broken = dict(VALID, fast_exit_steps=[0, 1])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_negative_fast_exit_step_raises(self):
        broken = dict(VALID, fast_exit_steps=[-1])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_duplicate_fast_exit_steps_raises(self):
        """Regression: dubbletter snedvred tidigare det aggregerade resultatet tyst."""
        broken = dict(VALID, fast_exit_steps=[1, 3, 3, 5])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_duplicate_twins_raises(self):
        broken = dict(VALID, twins=["twin_a", "twin_a"])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_path_traversal_strategy_name_raises(self):
        """Regression: strategy_name användes direkt för att bygga en
        filsökväg utan sanering, vilket möjliggjorde path traversal."""
        broken = dict(VALID, strategy_name="../../../../tmp/evil")
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_absolute_path_strategy_name_raises(self):
        broken = dict(VALID, strategy_name="/etc/evil")
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_unsafe_twin_name_raises(self):
        broken = dict(VALID, twins=["../evil"])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_bool_seed_raises(self):
        broken = dict(VALID, seed=True)
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_empty_fast_exit_steps_raises(self):
        broken = dict(VALID, fast_exit_steps=[])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)

    def test_empty_twins_raises(self):
        broken = dict(VALID, twins=[])
        with self.assertRaises(ConfigError):
            validate_and_normalize(broken)


if __name__ == "__main__":
    unittest.main()
