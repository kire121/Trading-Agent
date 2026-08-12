# Ny testsvit för lib/delivery.py:s leveranskvitto-hårdvakt
# (commit_sha/branch), tillagd vid Flodmärkets levande-komponenter-
# promovering (docs/INSTRUKTION.md avsnitt 3/7) -- se lib/delivery.py:s
# header för varför: Timglasets och Flodmärkets egna leveranser saknade
# commit-SHA helt.
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib import delivery  # noqa: E402
from lib.hashutil import compute_config_hash  # noqa: E402

VALID_SHA = "a" * 40


def _minimal_config():
    return {"strategy_name": "test_strategi", "seed": 1}


def _minimal_results(config):
    return {
        "config_hash": compute_config_hash(config),
        "data_window": {"oos_unlocked": False},
        "twins": [],
        "fast_exit_steps": [],
    }


class DeliverRequiresCommitShaAndBranchTests(unittest.TestCase):
    def test_missing_commit_sha_raises_and_writes_nothing(self):
        config = _minimal_config()
        results = _minimal_results(config)
        with tempfile.TemporaryDirectory() as tmp:
            strategy_dir = Path(tmp) / "test_strategi"
            with self.assertRaises(delivery.DeliveryError):
                delivery.deliver(strategy_dir, config, results, commit_sha=None, branch="main")
            self.assertFalse(strategy_dir.exists())

    def test_abbreviated_sha_rejected(self):
        config = _minimal_config()
        results = _minimal_results(config)
        with tempfile.TemporaryDirectory() as tmp:
            strategy_dir = Path(tmp) / "test_strategi"
            with self.assertRaises(delivery.DeliveryError):
                delivery.deliver(strategy_dir, config, results, commit_sha="a" * 7, branch="main")
            self.assertFalse(strategy_dir.exists())

    def test_uppercase_sha_rejected(self):
        # git rev-parse HEAD ger alltid gemener -- en versal SHA är ett
        # tecken på att värdet inte kommer från git rev-parse, avvisas.
        config = _minimal_config()
        results = _minimal_results(config)
        with tempfile.TemporaryDirectory() as tmp:
            strategy_dir = Path(tmp) / "test_strategi"
            with self.assertRaises(delivery.DeliveryError):
                delivery.deliver(strategy_dir, config, results, commit_sha="A" * 40, branch="main")

    def test_empty_branch_rejected(self):
        config = _minimal_config()
        results = _minimal_results(config)
        with tempfile.TemporaryDirectory() as tmp:
            strategy_dir = Path(tmp) / "test_strategi"
            with self.assertRaises(delivery.DeliveryError):
                delivery.deliver(strategy_dir, config, results, commit_sha=VALID_SHA, branch="   ")
            self.assertFalse(strategy_dir.exists())

    def test_valid_commit_sha_and_branch_succeeds_and_is_recorded(self):
        config = _minimal_config()
        results = _minimal_results(config)
        with tempfile.TemporaryDirectory() as tmp:
            strategy_dir = Path(tmp) / "test_strategi"
            assertions = delivery.deliver(strategy_dir, config, results,
                                           commit_sha=VALID_SHA, branch="claude/some-branch")
            self.assertTrue((strategy_dir / "results.json").exists())
            with open(strategy_dir / "results.json", encoding="utf-8") as f:
                written = json.load(f)
            self.assertEqual(written["delivery"]["commit_sha"], VALID_SHA)
            self.assertEqual(written["delivery"]["branch"], "claude/some-branch")

            names = {a["name"] for a in assertions}
            self.assertIn("leveranskvitto_har_commit_sha", names)
            self.assertIn("leveranskvitto_har_branch", names)
            by_name = {a["name"]: a for a in assertions}
            self.assertEqual(by_name["leveranskvitto_har_commit_sha"]["status"], "PASS")
            self.assertEqual(by_name["leveranskvitto_har_branch"]["status"], "PASS")


class BuildAssertionsCommitShaRowTests(unittest.TestCase):
    def test_missing_delivery_key_yields_fail_rows_not_a_crash(self):
        # build_assertions är en ren funktion -- kan anropas direkt utan
        # deliver()'s hårda vakt. Ett resultat utan "delivery"-nyckel ska
        # ge explicita FAIL-rader, aldrig en KeyError eller en utelämnad rad.
        config = _minimal_config()
        results = _minimal_results(config)
        assertions = delivery.build_assertions(config, results)
        by_name = {a["name"]: a for a in assertions}
        self.assertEqual(by_name["leveranskvitto_har_commit_sha"]["status"], "FAIL")
        self.assertEqual(by_name["leveranskvitto_har_branch"]["status"], "FAIL")


class CurrentCommitShaAndBranchTests(unittest.TestCase):
    def test_current_commit_sha_matches_git_rev_parse(self):
        import subprocess
        expected = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                   check=True).stdout.strip()
        self.assertEqual(delivery.current_commit_sha(), expected)
        self.assertTrue(delivery._COMMIT_SHA_RE.match(delivery.current_commit_sha()))

    def test_current_branch_matches_git_rev_parse(self):
        import subprocess
        expected = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                   text=True, check=True).stdout.strip()
        self.assertEqual(delivery.current_branch(), expected)


if __name__ == "__main__":
    unittest.main()
