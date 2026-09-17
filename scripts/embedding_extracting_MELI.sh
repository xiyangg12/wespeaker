#!/usr/bin/env bash

#SBATCH --job-name=wespeaker_meli
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=embedding_extracting_MELI_%j.out

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/embedding_extracting_MELI.sh [MODEL_NAME ...]

With no MODEL_NAME arguments, all 18 original pretrained models are used.

Environment overrides:
  WESPEAKER_REPO_ROOT       Repository checkout
                            (default: /home/xiyali/git/wespeaker)
  WESPEAKER_MELI_ROOT       Dataset root containing female/ and male/
                            (default: /home/xiyali/data/MELI)
  WESPEAKER_MODEL_ROOT      Directory containing the original model folders
                            (default: <repository>/models)
  WESPEAKER_OUTPUT_ROOT     Output directory
                            (default: <dataset>/embeddings_original_18)
  WESPEAKER_WAV_SCP         Existing manifest; if unset, one is generated
  WESPEAKER_CONDA_ENV       Conda environment (default: wespeaker)
  WESPEAKER_DEVICE          PyTorch device (default: cuda:0)
  WESPEAKER_CHECKPOINT      Original checkpoint filename
                            (default: avg_model.pt)
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

DEFAULT_REPO_ROOT=/home/xiyali/git/wespeaker
if [[ -n "${WESPEAKER_REPO_ROOT:-}" ]]; then
  REPO_ROOT=$WESPEAKER_REPO_ROOT
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && \
        -f "$SLURM_SUBMIT_DIR/scripts/create_wav_scp.py" ]]; then
  REPO_ROOT=$SLURM_SUBMIT_DIR
else
  REPO_ROOT=$DEFAULT_REPO_ROOT
fi

DATASET_ROOT=${WESPEAKER_MELI_ROOT:-/home/xiyali/data/MELI}
MODEL_ROOT=${WESPEAKER_MODEL_ROOT:-$REPO_ROOT/models}
OUTPUT_ROOT=${WESPEAKER_OUTPUT_ROOT:-$DATASET_ROOT/embeddings_original_18}
CONDA_ENV=${WESPEAKER_CONDA_ENV:-wespeaker}
DEVICE=${WESPEAKER_DEVICE:-cuda:0}
CHECKPOINT_NAME=${WESPEAKER_CHECKPOINT:-avg_model.pt}

DEFAULT_MODELS=(
  voxceleb_resnet34
  voxceleb_resnet34_LM
  voxceleb_resnet152_LM
  voxceleb_resnet221_LM
  voxceleb_resnet293_LM
  cnceleb_resnet34
  cnceleb_resnet34_LM
  voxblink2_samresnet34
  voxblink2_samresnet34_ft
  voxblink2_samresnet100
  voxblink2_samresnet100_ft
  voxceleb_ECAPA1024
  voxceleb_ECAPA1024_LM
  voxceleb_ECAPA512
  voxceleb_ECAPA512_LM
  voxceleb_ecapa512_dino
  voxceleb_CAM++
  voxceleb_CAM++_LM
)

if (( $# > 0 )); then
  MODEL_NAMES=("$@")
else
  MODEL_NAMES=("${DEFAULT_MODELS[@]}")
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "WeSpeaker repository root does not exist: $REPO_ROOT" >&2
  exit 1
fi
if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "MELI dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -d "$MODEL_ROOT" ]]; then
  echo "Original model root does not exist: $MODEL_ROOT" >&2
  exit 1
fi

REPO_ROOT=$(cd -- "$REPO_ROOT" && pwd)
DATASET_ROOT=$(cd -- "$DATASET_ROOT" && pwd)
MODEL_ROOT=$(cd -- "$MODEL_ROOT" && pwd)
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd)

SOURCE_SCP=${WESPEAKER_WAV_SCP:-$OUTPUT_ROOT/meli.wav.scp}
if [[ -z "${WESPEAKER_WAV_SCP:-}" ]]; then
  scan_directories=()
  for directory in female male; do
    if [[ -d "$DATASET_ROOT/$directory" ]]; then
      scan_directories+=("$directory")
    fi
  done
  if (( ${#scan_directories[@]} == 0 )); then
    echo "Neither female/ nor male/ exists under $DATASET_ROOT" >&2
    exit 1
  fi
  python3 "$REPO_ROOT/scripts/create_wav_scp.py" \
    "$DATASET_ROOT" "$SOURCE_SCP" "${scan_directories[@]}"
fi

if [[ ! -f "$SOURCE_SCP" ]]; then
  echo "SCP manifest does not exist: $SOURCE_SCP" >&2
  exit 1
fi

expected_count=0
duplicate_utterance=$(awk 'NF && seen[$1]++ { print $1; exit }' "$SOURCE_SCP")
if [[ -n "$duplicate_utterance" ]]; then
  echo "Duplicate utterance ID in manifest: $duplicate_utterance" >&2
  exit 1
fi

while read -r utterance relative_audio; do
  [[ -n "$utterance" ]] || continue
  if [[ ! -f "$DATASET_ROOT/$relative_audio" ]]; then
    echo "Missing audio for $utterance: $DATASET_ROOT/$relative_audio" >&2
    exit 1
  fi
  ((expected_count += 1))
done < "$SOURCE_SCP"

if (( expected_count == 0 )); then
  echo "No utterances found in $SOURCE_SCP" >&2
  exit 1
fi

for model_name in "${MODEL_NAMES[@]}"; do
  checkpoint="$MODEL_ROOT/$model_name/$CHECKPOINT_NAME"
  config="$MODEL_ROOT/$model_name/config.yaml"
  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing original checkpoint for $model_name: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -f "$config" ]]; then
    echo "Missing original config for $model_name: $config" >&2
    exit 1
  fi
done

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "Conda is unavailable; activate the WeSpeaker environment first." >&2
  exit 1
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "Dataset:          $DATASET_ROOT"
echo "Manifest:         $SOURCE_SCP"
echo "Original models:  $MODEL_ROOT"
echo "Outputs:          $OUTPUT_ROOT"
echo "Device:           $DEVICE"
echo "Utterances:       $expected_count"
echo "Models:           ${#MODEL_NAMES[@]}"

for model_name in "${MODEL_NAMES[@]}"; do
  source_dir="$MODEL_ROOT/$model_name"
  checkpoint="$source_dir/$CHECKPOINT_NAME"
  config="$source_dir/config.yaml"
  bundle_dir="$OUTPUT_ROOT/.model_bundles/$model_name"
  model_output_dir="$OUTPUT_ROOT/$model_name"

  mkdir -p "$bundle_dir" "$model_output_dir"
  ln -sfn "$config" "$bundle_dir/config.yaml"
  # This branch's Speaker loader expects the checkpoint to be named model_5.pt.
  ln -sfn "$checkpoint" "$bundle_dir/model_5.pt"

  echo "[$model_name] extracting $expected_count embeddings"
  (
    cd -- "$DATASET_ROOT"
    python3 -m wespeaker.cli.speaker \
      --task embedding_kaldi \
      --pretrain "$bundle_dir" \
      --wav_scp "$SOURCE_SCP" \
      --output_file "$model_output_dir/embeddings" \
      --device "$DEVICE" \
      --resample_rate 16000
  )

  if [[ ! -s "$model_output_dir/embeddings.ark" || \
        ! -s "$model_output_dir/embeddings.scp" ]]; then
    echo "Inference did not create both output files for $model_name" >&2
    exit 1
  fi
  echo "[$model_name] wrote $model_output_dir/embeddings.{ark,scp}"
done

echo "Completed MELI embedding extraction for ${#MODEL_NAMES[@]} model(s)."
