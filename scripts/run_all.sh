#!/usr/bin/env bash
# Runs the experiment matrix (CONTRACTS C5, Phase 08):
#   bash scripts/run_all.sh [--models ...] [--strategies ...] [--seeds ...]
#
# Skips evaluated runs, resumes unfinished ones and evaluates finished ones, so it can be started
# again any number of times. Order: seed 0, then 1, then 2, cheapest first within a seed.
#
# Environment:
#   SESSION_BUDGET_MIN  whole minutes this call may use. A run stops at the last epoch that fits,
#                       estimated from its last epoch time (scripts/plan.py), and is left resumable
#                       with STOP_AFTER_EPOCHS, so that no training code changes. Exit code 0.
#   RESULTS_SYNC_DIR    folder (Drive, on Colab) to copy results into after each evaluated run.
#   PYTHON              interpreter, default `python`.
#
# Exit code 1: a run failed, or one needs a human. The matrix stops at once, because the same fault
# would hit the next runs too.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
unset STOP_AFTER_EPOCHS

PY="${PYTHON:-python}"
RUNS_DIR="${RUNS_DIR:-./runs}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
SYNC_DIR="${RESULTS_SYNC_DIR:-}"
filters=()
while (($#)); do
    case "$1" in
    --models | --strategies | --seeds)
        filters+=("$1")
        shift
        while (($#)) && [[ "$1" != --* ]]; do
            filters+=("$1")
            shift
        done
        ;;
    *)
        echo "unknown argument: $1" >&2
        exit 2
        ;;
    esac
done

deadline=""
if [[ -n "${SESSION_BUDGET_MIN:-}" ]]; then
    if [[ ! "$SESSION_BUDGET_MIN" =~ ^[0-9]+$ ]]; then
        echo "SESSION_BUDGET_MIN must be whole minutes, got: $SESSION_BUDGET_MIN" >&2
        exit 2
    fi
    deadline=$(($(date +%s) + SESSION_BUDGET_MIN * 60))
fi

# Copy to a hidden name, then rename, so that the destination never holds half a file.
sync_file() {
    local source="$1" dest="$2"
    [[ -f "$source" ]] || return 0
    if cmp -s "$source" "$dest"; then
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    local partial
    partial="$(dirname "$dest")/.$(basename "$dest").partial"
    cp "$source" "$partial"
    mv "$partial" "$dest"
}

# The CSV goes last: its presence means the whole evaluation is saved.
sync_run() {
    [[ -n "$SYNC_DIR" ]] || return 0
    local run_id="$1" name
    for name in "${run_id}_train_log.csv" "${run_id}_confusion.npz" "${run_id}.csv"; do
        sync_file "$RESULTS_DIR/raw/$name" "$SYNC_DIR/raw/$name"
    done
    for name in run_meta.json config_resolved.yaml train_log.csv; do
        sync_file "$RUNS_DIR/$run_id/$name" "$SYNC_DIR/run_folders/$run_id/$name"
    done
    sync_file "$RESULTS_DIR/runs.csv" "$SYNC_DIR/runs.csv"
}

finish() {
    local code="$1" run_id="${2:-}"
    "$PY" -m scripts.registry || code=1
    if [[ -n "$run_id" ]]; then
        sync_run "$run_id" || code=1
    fi
    "$PY" -m scripts.plan "${filters[@]}" || code=1
    exit "$code"
}

train_run() {
    local run_id="$1" stop_after="$2" short strategy seed_tag
    IFS=- read -r short strategy seed_tag <<<"$run_id"
    local command=("$PY" -m src.train --config "configs/${short}_${strategy}.yaml" --seed "${seed_tag#s}")
    if [[ "$stop_after" == "-" ]]; then
        "${command[@]}"
    else
        STOP_AFTER_EPOCHS="$stop_after" "${command[@]}"
    fi
}

# No CSV here means no finished evaluation, so a confusion file on its own is a leftover from one
# that was cut short, and only that is replaced.
evaluate_run() {
    local run_id="$1" force=()
    if [[ -f "$RESULTS_DIR/raw/${run_id}_confusion.npz" ]]; then
        force=(--force)
    fi
    "$PY" -m src.evaluate --run-dir "$RUNS_DIR/$run_id" "${force[@]}"
}

# Run one id until it is evaluated. Returns only if there is nothing more to do for it.
advance_run() {
    local run_id="$1" previous_done="" step action detail stop_after done_epochs
    local args=(--next-step "$run_id")
    while true; do
        if [[ -n "$deadline" ]]; then
            step="$("$PY" -m scripts.plan "${args[@]}" --remaining-s "$((deadline - $(date +%s)))")"
        else
            step="$("$PY" -m scripts.plan "${args[@]}")"
        fi
        read -r action detail <<<"$step"
        case "$action" in
        skip)
            echo "$run_id: already evaluated, skipping"
            sync_run "$run_id"
            return 0
            ;;
        evaluate)
            echo "$run_id: evaluating"
            evaluate_run "$run_id" || {
                echo "ERROR: evaluation of $run_id failed" >&2
                finish 1 "$run_id"
            }
            "$PY" -m scripts.registry
            sync_run "$run_id"
            ;;
        train)
            read -r stop_after done_epochs <<<"$detail"
            if [[ "$done_epochs" == "$previous_done" ]]; then
                echo "ERROR: $run_id made no progress past epoch $done_epochs" >&2
                finish 1 "$run_id"
            fi
            previous_done="$done_epochs"
            echo "$run_id: training from epoch $done_epochs, stop after: $stop_after"
            train_run "$run_id" "$stop_after" || {
                echo "ERROR: training of $run_id failed" >&2
                finish 1 "$run_id"
            }
            ;;
        stop)
            echo "session budget reached before $run_id: $detail"
            finish 0 "$run_id"
            ;;
        *)
            echo "ERROR: $step" >&2
            finish 1
            ;;
        esac
    done
}

run_ids="$("$PY" -m scripts.plan --run-ids "${filters[@]}")"
for run_id in $run_ids; do
    advance_run "$run_id"
done
echo "all requested runs are evaluated"
finish 0
