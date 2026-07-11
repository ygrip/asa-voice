#!/usr/bin/env python3
"""Verify Settings, runtime references, and .env.example without importing model dependencies."""
import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "app" / "config.py"
ENV_EXAMPLE = ROOT / ".env.example"
REMOVED_KEYS = {"STT_BEAM_SIZE", "STT_BEST_OF", "STT_PROMPT", "STT_VAD_FILTER"}


def setting_names() -> set[str]:
    tree = ast.parse(CONFIG.read_text(), filename=str(CONFIG))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Settings":
            return {
                child.target.id
                for child in node.body
                if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name)
            }
    raise RuntimeError("Settings class not found")


def env_keys() -> set[str]:
    keys: set[str] = set()
    for raw_line in ENV_EXAMPLE.read_text().splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#") and "=" in line:
            keys.add(line.split("=", 1)[0])
    return keys


def runtime_references() -> set[str]:
    references: set[str] = set()
    for source in (ROOT / "app").rglob("*.py"):
        if source == CONFIG:
            continue
        tree = ast.parse(source.read_text(), filename=str(source))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "settings"
            ):
                references.add(node.attr)
    return references


def main() -> int:
    settings = setting_names()
    expected_env = {name.upper() for name in settings}
    actual_env = env_keys()
    errors: list[str] = []

    if expected_env != actual_env:
        missing = sorted(expected_env - actual_env)
        extra = sorted(actual_env - expected_env)
        if missing:
            errors.append("Missing .env.example keys: " + ", ".join(missing))
        if extra:
            errors.append("Unknown .env.example keys: " + ", ".join(extra))

    unused = sorted((settings - {"app_name"}) - runtime_references())
    if unused:
        errors.append("Unused Settings fields: " + ", ".join(unused))

    removed = sorted(REMOVED_KEYS & actual_env)
    if removed:
        errors.append("Removed environment keys returned: " + ", ".join(removed))

    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    print("sidecar configuration contract is synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
