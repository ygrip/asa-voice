#!/usr/bin/env python3
"""Assert the hosted image excludes all local-ML dependencies (setara-nx07.5, plan §13).

Run inside the built hosted image, e.g.:
    docker run --rm "$IMAGE" python scripts/assert_hosted_deps_absent.py
"""
import importlib.util
import sys

FORBIDDEN_PACKAGES = ("torch", "pocket_tts", "faster_whisper", "ctranslate2", "scipy")


def main() -> int:
    present = [pkg for pkg in FORBIDDEN_PACKAGES if importlib.util.find_spec(pkg) is not None]
    if present:
        print(f"error: hosted image must not contain: {', '.join(present)}", file=sys.stderr)
        return 1
    print(f"ok: none of {', '.join(FORBIDDEN_PACKAGES)} present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
