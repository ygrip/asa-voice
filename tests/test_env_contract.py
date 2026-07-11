"""Configuration contract: runtime settings and the deployable env template stay synchronized."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_configuration_contract() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_config_contract.py")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
