#!/usr/bin/env bash

#SBATCH --job-name=wespeaker_infer
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/wespeaker_infer_%j.out

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/infer_perception_study_14.sh \
    DATASET_ROOT FINETUNED_MODEL_ROOT OUTPUT_ROOT [MODEL_NAME ...]

Arguments:
  DATASET_ROOT           Cluster directory containing the audio paths named in
                         the selected SCP manifest.
  FINETUNED_MODEL_ROOT   Path containing the eleven fine-tuned experiment trees.
                         This is normally the repository's exp/ directory.
  OUTPUT_ROOT            Directory in which per-model .ark/.scp files are saved.
  MODEL_NAME             Optional model names. If omitted, all 18 paper models run.

Environment overrides:
  WESPEAKER_CONDA_ENV            Conda environment name (default: wespeaker)
  WESPEAKER_DEVICE               PyTorch device (default: cuda:0)
  WESPEAKER_CHECKPOINT           Fine-tuned checkpoint name (default: model_5.pt)
  WESPEAKER_PRETRAINED_ROOT      Root containing ECAPA/CAM++ model directories
                                 (default: repository root)
  WESPEAKER_PRETRAINED_CHECKPOINT  Pretrained checkpoint name
                                 (default: avg_model.pt)
  WESPEAKER_REPO_ROOT            WeSpeaker checkout path. Normally detected
                                 from SLURM_SUBMIT_DIR when submitted by Slurm.
  WESPEAKER_WAV_SCP              SCP manifest to use. May be an absolute path
                                 or a path relative to the repository. Defaults
                                 to manifests/perception_study_14.wav.scp.

Example:
  mkdir -p logs
  sbatch scripts/infer_perception_study_14.sh \
    /cluster/data/perception_study_14 \
    /cluster/home/me/git/wespeaker/exp \
    /cluster/results/perception_study_14
EOF
}

if (( $# < 3 )); then
  usage >&2
  exit 2
fi

DATASET_ROOT=$1
FINETUNED_MODEL_ROOT=$2
OUTPUT_ROOT=$3
shift 3

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST_RELATIVE_PATH=manifests/perception_study_14.wav.scp
FINETUNED_MODEL_PARENT=$(dirname -- "$FINETUNED_MODEL_ROOT")
if [[ -n "${WESPEAKER_REPO_ROOT:-}" ]]; then
  REPO_ROOT=$WESPEAKER_REPO_ROOT
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && \
        -f "$SLURM_SUBMIT_DIR/$MANIFEST_RELATIVE_PATH" ]]; then
  # sbatch executes a spool copy under /var/spool/slurmd, so BASH_SOURCE does
  # not identify the checkout. SLURM_SUBMIT_DIR is the directory from which
  # sbatch was invoked.
  REPO_ROOT=$SLURM_SUBMIT_DIR
elif [[ -f "$FINETUNED_MODEL_PARENT/$MANIFEST_RELATIVE_PATH" ]]; then
  # The normal second argument is <repository>/exp, making its parent another
  # reliable way to recover the checkout when sbatch was launched elsewhere.
  REPO_ROOT=$FINETUNED_MODEL_PARENT
else
  REPO_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
fi

if [[ ! -d "$REPO_ROOT" ]]; then
  echo "WeSpeaker repository root does not exist: $REPO_ROOT" >&2
  exit 1
fi
REPO_ROOT=$(cd -- "$REPO_ROOT" && pwd)
SOURCE_SCP=${WESPEAKER_WAV_SCP:-$REPO_ROOT/$MANIFEST_RELATIVE_PATH}
if [[ "$SOURCE_SCP" != /* ]]; then
  SOURCE_SCP="$REPO_ROOT/$SOURCE_SCP"
fi

CONDA_ENV=${WESPEAKER_CONDA_ENV:-wespeaker}
DEVICE=${WESPEAKER_DEVICE:-cuda:0}
FINETUNED_CHECKPOINT=${WESPEAKER_CHECKPOINT:-model_5.pt}
PRETRAINED_MODEL_ROOT=${WESPEAKER_PRETRAINED_ROOT:-$REPO_ROOT}
PRETRAINED_CHECKPOINT=${WESPEAKER_PRETRAINED_CHECKPOINT:-avg_model.pt}

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

PRETRAINED_MODELS=(
  voxceleb_ECAPA1024
  voxceleb_ECAPA1024_LM
  voxceleb_ECAPA512
  voxceleb_ECAPA512_LM
  voxceleb_ecapa512_dino
  voxceleb_CAM++
  voxceleb_CAM++_LM
)

declare -A IS_PRETRAINED_MODEL=()
for model_name in "${PRETRAINED_MODELS[@]}"; do
  IS_PRETRAINED_MODEL[$model_name]=1
done

if (( $# > 0 )); then
  MODEL_NAMES=("$@")
else
  MODEL_NAMES=("${DEFAULT_MODELS[@]}")
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "Dataset root does not exist: $DATASET_ROOT" >&2
  exit 1
fi
if [[ ! -d "$FINETUNED_MODEL_ROOT" ]]; then
  echo "Fine-tuned model root does not exist: $FINETUNED_MODEL_ROOT" >&2
  exit 1
fi
if [[ ! -f "$SOURCE_SCP" ]]; then
  echo "SCP manifest does not exist: $SOURCE_SCP" >&2
  exit 1
fi

# The manifest deliberately uses paths relative to DATASET_ROOT so that it is
# portable between the local machine and the cluster.
expected_count=0
declare -A seen_utterances=()
while read -r utterance relative_audio; do
  [[ -n "$utterance" ]] || continue
  if [[ -n "${seen_utterances[$utterance]:-}" ]]; then
    echo "Duplicate utterance ID in manifest: $utterance" >&2
    exit 1
  fi
  seen_utterances[$utterance]=1
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

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [[ -f "$HOME/miniconda3/bin/activate" ]]; then
  # Compatibility with the activation style used by the original cluster job.
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/bin/activate" "$CONDA_ENV"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV"
else
  echo "Conda is unavailable; activate the WeSpeaker environment first." >&2
  exit 1
fi

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT=$(cd -- "$OUTPUT_ROOT" && pwd)
DATASET_ROOT=$(cd -- "$DATASET_ROOT" && pwd)
FINETUNED_MODEL_ROOT=$(cd -- "$FINETUNED_MODEL_ROOT" && pwd)
if [[ -d "$PRETRAINED_MODEL_ROOT" ]]; then
  PRETRAINED_MODEL_ROOT=$(cd -- "$PRETRAINED_MODEL_ROOT" && pwd)
fi

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

echo "Dataset:   $DATASET_ROOT"
echo "Fine-tuned models: $FINETUNED_MODEL_ROOT"
echo "Pretrained models: $PRETRAINED_MODEL_ROOT"
echo "Outputs:   $OUTPUT_ROOT"
echo "Device:    $DEVICE"
echo "Utterances: $expected_count"

for model_name in "${MODEL_NAMES[@]}"; do
  if [[ -n "${IS_PRETRAINED_MODEL[$model_name]:-}" ]]; then
    model_kind=pretrained
    source_dir="$PRETRAINED_MODEL_ROOT/$model_name"
    if [[ ! -d "$source_dir" && -d "$PRETRAINED_MODEL_ROOT/models/$model_name" ]]; then
      source_dir="$PRETRAINED_MODEL_ROOT/models/$model_name"
    fi
    checkpoint="$source_dir/$PRETRAINED_CHECKPOINT"
    config="$source_dir/config.yaml"
  else
    model_kind=fine-tuned
    experiment_dir="$FINETUNED_MODEL_ROOT/$model_name"
    checkpoint="$experiment_dir/models/$FINETUNED_CHECKPOINT"
    config="$experiment_dir/config.yaml"
    if [[ ! -f "$config" && -f "$experiment_dir/models/config.yaml" ]]; then
      config="$experiment_dir/models/config.yaml"
    fi
  fi

  if [[ ! -f "$checkpoint" ]]; then
    echo "Missing $model_kind checkpoint for $model_name: $checkpoint" >&2
    exit 1
  fi
  if [[ ! -f "$config" ]]; then
    echo "Missing $model_kind config for $model_name: $config" >&2
    exit 1
  fi

  # This fork's Speaker loader requires config.yaml and model_5.pt in the same
  # directory. Make a lightweight symlink bundle without altering exp/.
  bundle_dir="$OUTPUT_ROOT/.model_bundles/$model_name"
  model_output_dir="$OUTPUT_ROOT/$model_name"
  mkdir -p "$bundle_dir" "$model_output_dir"
  ln -sfn "$config" "$bundle_dir/config.yaml"
  ln -sfn "$checkpoint" "$bundle_dir/model_5.pt"

  echo "[$model_name] extracting $expected_count embeddings from $model_kind model"
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

echo "Completed inference for ${#MODEL_NAMES[@]} model(s)."
