#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${1:?usage: convert_whisper_ct2.sh MODEL_ID OUTPUT_DIR QUANTIZATION REVISION}"
OUTPUT_DIR="${2:?usage: convert_whisper_ct2.sh MODEL_ID OUTPUT_DIR QUANTIZATION REVISION}"
QUANTIZATION="${3:-int8}"
REVISION="${4:?a pinned Hugging Face commit revision is required}"

if [[ "${REVISION}" == "main" ]]; then
  echo "Refusing mutable revision 'main'; pass a commit SHA" >&2
  exit 2
fi

if [[ -f "${OUTPUT_DIR}/.asa_model_ready" ]]; then
  echo "Model artifact already ready: ${OUTPUT_DIR}"
  exit 0
fi

tmp_dir="${OUTPUT_DIR}.tmp.$$"
cleanup() {
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT
mkdir -p "${tmp_dir}"

ct2-transformers-converter \
  --model "${MODEL_ID}" \
  --revision "${REVISION}" \
  --output_dir "${tmp_dir}" \
  --copy_files tokenizer.json preprocessor_config.json generation_config.json \
  --quantization "${QUANTIZATION}"

printf '%s\n' \
  "model=${MODEL_ID}" \
  "revision=${REVISION}" \
  "quantization=${QUANTIZATION}" > "${tmp_dir}/artifact.properties"
touch "${tmp_dir}/.asa_model_ready"
rm -rf "${OUTPUT_DIR}"
mv -f "${tmp_dir}" "${OUTPUT_DIR}"
trap - EXIT

echo "Converted ${MODEL_ID}@${REVISION} to ${OUTPUT_DIR} (${QUANTIZATION})"
