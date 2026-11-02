#!/usr/bin/env bash
# MAX78002 toolchain for the Group 17 ECG model. Run inside WSL/Linux:
#   bash max78002/run_toolchain.sh install|train|quantize|evaluate|synth|all
# AI8X points at the folder holding ai8x-training/ and ai8x-synthesis/;
# VENV at a Python environment with both projects' requirements.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG="$REPO/max78002/model_configuration"
AI8X="${AI8X:-/root/max78002}"
VENV="${VENV:-$AI8X/.venv}"
TRAIN="$AI8X/ai8x-training"
SYNTH="$AI8X/ai8x-synthesis"
NAME=ecg_b2
QCKPT="$SYNTH/trained/ai87-ecg-b2-qat8-q.pth.tar"
# torch.compile is unavailable on Python 3.12; disabling it changes speed, not results
COMMON=(--model ai87ecgnet --device MAX78002 --use-bias --confusion --compiler-mode none)

# shellcheck disable=SC1091
source "$VENV/bin/activate"

install() {
    cp "$CFG/ai87ecgnet.py" "$TRAIN/models/"
    cp "$CFG/ecg.py" "$TRAIN/datasets/"
    cp "$CFG/qat_policy_ecg.yaml" "$CFG/schedule_ecg.yaml" "$TRAIN/policies/"
    cp "$CFG/ecg-b2.yaml" "$SYNTH/networks/"
    mkdir -p "$TRAIN/data/ECG"
    cp "$REPO/data/processed/beats.npz" "$TRAIN/data/ECG/"
    echo "installed into $AI8X"
}

train() {
    cd "$TRAIN"
    python train.py --epochs "${EPOCHS:-100}" --optimizer Adam --lr 0.001 --wd 0 \
        --batch-size 256 --deterministic --workers 4 \
        --compress policies/schedule_ecg.yaml --qat-policy policies/qat_policy_ecg.yaml \
        --dataset ECG --name "$NAME" "${COMMON[@]}" "$@"
    local log
    log="$(ls -td logs/*"$NAME"* | head -1)"
    mkdir -p "$REPO/max78002/quantization_output/training_log"
    cp "$log"/*.log "$log/${NAME}_qat_best.pth.tar" \
        "$REPO/max78002/quantization_output/training_log/"
    echo "training log: $TRAIN/$log"
}

quantize() {
    cd "$SYNTH"
    python quantize.py "$REPO/max78002/quantization_output/training_log/${NAME}_qat_best.pth.tar" \
        "$QCKPT" --device MAX78002 -v 2>&1 | tee "$REPO/max78002/quantization_output/quantize.log"
    cp "$QCKPT" "$REPO/max78002/quantization_output/"
}

evaluate() {
    # validation records only; DS2 is evaluated once, later, by the evaluation task
    cd "$TRAIN"
    python train.py --evaluate --exp-load-weights-from "$QCKPT" -8 --dataset ECG \
        --save-sample 10 "${COMMON[@]}" 2>&1 \
        | tee "$REPO/max78002/quantization_output/evaluate_int8_val.log"
    cp sample_ecg.npy "$SYNTH/tests/sample_ecg.npy"
    cp sample_ecg.npy "$REPO/max78002/model_configuration/"
}

synth() {
    cd "$SYNTH"
    # synthesise outside /mnt/c: the Windows path contains spaces
    rm -rf "$AI8X/out/$NAME"
    python ai8xize.py --no-version-check --verbose --test-dir "$AI8X/out" --prefix "$NAME" \
        --checkpoint-file "$QCKPT" --config-file networks/ecg-b2.yaml \
        --sample-input tests/sample_ecg.npy --device MAX78002 \
        --softmax --compact-data --mexpress --timer 0 --display-checkpoint
    rm -rf "$REPO/max78002/synthesis_output/$NAME"
    mkdir -p "$REPO/max78002/synthesis_output"
    cp -r "$AI8X/out/$NAME" "$REPO/max78002/synthesis_output/"
    grep -E "memory:" "$AI8X/out/$NAME/log.txt" || true
}

stage="${1:-all}"
shift || true
case "$stage" in
    install|train|quantize|evaluate|synth) "$stage" "$@" ;;
    all) install; train "$@"; quantize; evaluate; synth ;;
    *) echo "unknown stage: $stage" >&2; exit 2 ;;
esac
