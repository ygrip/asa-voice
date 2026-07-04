from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_conversion_requires_pinned_revision_and_writes_atomically() -> None:
    source = (ROOT / "scripts" / "convert_whisper_ct2.sh").read_text()

    assert "a pinned Hugging Face commit revision is required" in source
    assert "Refusing mutable revision 'main'" in source
    assert '--revision "${REVISION}"' in source
    assert 'tmp_dir="${OUTPUT_DIR}.tmp.$$"' in source
    assert 'touch "${tmp_dir}/.asa_model_ready"' in source
    assert 'mv -f "${tmp_dir}" "${OUTPUT_DIR}"' in source


def test_conversion_dependencies_are_separate_from_runtime() -> None:
    runtime = (ROOT / "requirements.txt").read_text()
    conversion = (ROOT / "requirements-convert.txt").read_text()

    assert "transformers[torch]" not in runtime
    assert "transformers[torch]==" in conversion
    assert "ctranslate2==" in conversion
