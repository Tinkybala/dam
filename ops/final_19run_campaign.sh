#!/usr/bin/env bash
set -u

: "${FINAL_RUN_COMMIT:?FINAL_RUN_COMMIT is required}"
: "${RELEASE_DIR:?RELEASE_DIR is required}"
: "${PYTHON_BIN:?PYTHON_BIN is required}"

final_run_commit="$FINAL_RUN_COMMIT"
release_dir="$RELEASE_DIR"
python_bin="$PYTHON_BIN"

cd "$release_dir" || exit 2
mkdir -p results/final/logs results/final/status

lock_dir='results/final/.campaign-lock'
if ! mkdir "$lock_dir" 2>/dev/null; then
  echo "STOP: campaign lock already exists: $lock_dir"
  exit 90
fi

gpu_monitor_pid=''

cleanup() {
  if [ -n "$gpu_monitor_pid" ]; then
    kill "$gpu_monitor_pid" 2>/dev/null || true
    wait "$gpu_monitor_pid" 2>/dev/null || true
  fi
  rmdir "$lock_dir" 2>/dev/null || true
}

trap cleanup EXIT
trap 'exit 130' INT TERM

resolve_output_dir() {
  "$python_bin" -c 'import pathlib,sys,yaml; p=pathlib.Path(sys.argv[1]); c=yaml.safe_load(p.read_text(encoding="utf-8")); print((p.parent / c["output_dir"]).resolve())' "$1"
}

validate_metrics_schema() {
  metrics_path="$1"
  expected_model="$2"
  "$python_bin" -c 'import json,math,sys; p,commit,model=sys.argv[1:4]; d=json.load(open(p,encoding="utf-8")); assert d.get("commit")==commit; assert d.get("model")==model; assert d.get("selected_user_count")==60384; assert d.get("config",{}).get("evaluate_test") is True; assert d.get("validation",{}).get("user_count")==60384; assert d.get("test",{}).get("user_count")==60384; assert math.isfinite(float(d["validation"]["ndcg@10"])); assert math.isfinite(float(d["validation"]["hit_rate@10"])); assert math.isfinite(float(d["test"]["ndcg@10"])); assert math.isfinite(float(d["test"]["hit_rate@10"])); assert model=="popular" or (d.get("device")=="cuda" and d.get("gpu_sampling") is True); epochs=d.get("epochs_completed"); assert model=="popular" or (isinstance(epochs,int) and 1 <= epochs <= int(d["config"]["epochs"]))' "$metrics_path" "$final_run_commit" "$expected_model"
}

validate_config_gate() {
  config_path="$1"
  require_cuda="$2"
  "$python_bin" - "$config_path" "$require_cuda" <<'PY'
import sys
import yaml

path, require_cuda = sys.argv[1], sys.argv[2] == 'true'
with open(path, encoding='utf-8') as handle:
    config = yaml.safe_load(handle)

assert config.get('evaluate_test') is True, f'{path}: test is not enabled'
assert 'development_users_file' not in config, f'{path}: development subset is forbidden'
if require_cuda:
    assert config.get('device') == 'cuda', f'{path}: explicit CUDA is required'
    assert config.get('gpu_sampling') is True, f'{path}: GPU sampling is required'
PY
}

wait_for_gpu_attach() {
  launcher_pid="$1"
  config_path="$2"
  proof_path="$3"

  attempt=1
  while [ "$attempt" -le 24 ]; do
    if ! kill -0 "$launcher_pid" 2>/dev/null; then
      return 1
    fi

    train_pids="$(pgrep -f "python.*-m src.train --config $config_path" || true)"
    if [ -n "$train_pids" ]; then
      compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
      for train_pid in $train_pids; do
        if printf '%s\n' "$compute_pids" | grep -qx "$train_pid"; then
          {
            echo "gpu_attach=PASS"
            echo "python_pid=$train_pid"
            nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
          } > "$proof_path"
          return 0
        fi
      done
    fi

    sleep 5
    attempt=$((attempt + 1))
  done

  return 1
}

run_one() {
  gpu_id="$1"
  config_name="$2"
  run_name="${config_name%.yaml}"
  config_path="configs/final/$config_name"
  log_path="results/final/logs/$run_name.log"
  status_path="results/final/status/$run_name.status"
  proof_path="results/final/status/$run_name.gpu-proof"
  output_dir="$(resolve_output_dir "$config_path")" || return 81
  expected_model="$("$python_bin" -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1],encoding="utf-8"))["model"])' "$config_path")" || return 81

  if [ -e "$output_dir" ] || [ -e "$log_path" ] || [ -e "$status_path" ]; then
    echo "STOP: existing output/log/status for $run_name"
    return 82
  fi

  started_at="$(date --iso-8601=seconds)"
  echo "START run=$run_name gpu=$gpu_id time=$started_at"

  timeout 120m env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    SOURCE_COMMIT="$final_run_commit" \
    PYTHONPATH="$release_dir" \
    "$python_bin" -m src.train --config "$config_path" \
    > "$log_path" 2>&1 &
  launcher_pid=$!

  if ! wait_for_gpu_attach "$launcher_pid" "$config_path" "$proof_path"; then
    if kill -0 "$launcher_pid" 2>/dev/null; then
      echo "STOP: GPU attach was not proven for $run_name" >> "$log_path"
      kill "$launcher_pid" 2>/dev/null || true
    fi
    wait "$launcher_pid" 2>/dev/null || true
    printf 'run=%s\ngpu=%s\nstarted_at=%s\ngpu_gate=FAIL\ncommit=%s\n' \
      "$run_name" "$gpu_id" "$started_at" "$final_run_commit" > "$status_path"
    return 88
  fi

  wait "$launcher_pid"
  exit_code=$?
  ended_at="$(date --iso-8601=seconds)"

  printf 'run=%s\ngpu=%s\nstarted_at=%s\nended_at=%s\nexit_code=%s\ngpu_gate=PASS\ncommit=%s\n' \
    "$run_name" "$gpu_id" "$started_at" "$ended_at" "$exit_code" "$final_run_commit" \
    > "$status_path"

  if [ "$exit_code" -ne 0 ]; then
    echo "STOP: run failed or timed out: $run_name exit=$exit_code"
    return "$exit_code"
  fi

  metrics_path="$output_dir/metrics.json"
  if ! validate_metrics_schema "$metrics_path" "$expected_model"; then
    echo "STOP: metrics schema gate failed: $run_name"
    return 83
  fi

  printf 'schema_gate=PASS\n' >> "$status_path"
  echo "DONE run=$run_name gpu=$gpu_id time=$ended_at"
}

run_popular() {
  config_name='anime_popular_final.yaml'
  run_name="${config_name%.yaml}"
  config_path="configs/final/$config_name"
  log_path="results/final/logs/$run_name.log"
  status_path="results/final/status/$run_name.status"
  output_dir="$(resolve_output_dir "$config_path")" || return 81

  if [ -e "$output_dir" ] || [ -e "$log_path" ] || [ -e "$status_path" ]; then
    echo "STOP: existing output/log/status for $run_name"
    return 82
  fi

  started_at="$(date --iso-8601=seconds)"
  timeout 30m env SOURCE_COMMIT="$final_run_commit" PYTHONPATH="$release_dir" \
    "$python_bin" -m src.train --config "$config_path" > "$log_path" 2>&1
  exit_code=$?
  ended_at="$(date --iso-8601=seconds)"
  printf 'run=%s\ngpu=none\nstarted_at=%s\nended_at=%s\nexit_code=%s\ncommit=%s\n' \
    "$run_name" "$started_at" "$ended_at" "$exit_code" "$final_run_commit" > "$status_path"

  if [ "$exit_code" -ne 0 ]; then
    return "$exit_code"
  fi
  validate_metrics_schema "$output_dir/metrics.json" popular || return 83
  printf 'schema_gate=PASS\n' >> "$status_path"
}

run_worker() {
  gpu_id="$1"
  shift
  for config_name in "$@"; do
    if ! run_one "$gpu_id" "$config_name"; then
      echo "WORKER_STOP gpu=$gpu_id config=$config_name"
      return 1
    fi
  done
}

all_configs=(
  anime_popular_final.yaml
  anime_bpr_final_seed42.yaml
  anime_bpr_final_seed43.yaml
  anime_bpr_final_seed44.yaml
  anime_bpr_ensemble_component_final_seed42.yaml
  anime_bpr_ensemble_component_final_seed43.yaml
  anime_bpr_ensemble_component_final_seed44.yaml
  anime_gmf_final_seed42.yaml
  anime_gmf_final_seed43.yaml
  anime_gmf_final_seed44.yaml
  anime_mlp_final_seed42.yaml
  anime_mlp_final_seed43.yaml
  anime_mlp_final_seed44.yaml
  anime_neumf_final_seed42.yaml
  anime_neumf_final_seed43.yaml
  anime_neumf_final_seed44.yaml
  anime_weighted_neumf_final_seed42.yaml
  anime_weighted_neumf_final_seed43.yaml
  anime_weighted_neumf_final_seed44.yaml
)

if [ "${#all_configs[@]}" -ne 19 ]; then
  echo 'STOP: internal matrix is not 19 configs'
  exit 84
fi

actual_config_count="$(find configs/final -maxdepth 1 -type f -name '*.yaml' | wc -l | tr -d '[:space:]')"
if [ "$actual_config_count" -ne 19 ]; then
  echo "STOP: expected 19 final configs, found $actual_config_count"
  exit 84
fi

for config_name in "${all_configs[@]}"; do
  config_path="configs/final/$config_name"
  if [ ! -f "$config_path" ]; then
    echo "STOP: missing config: $config_path"
    exit 84
  fi
  if ! validate_config_gate "$config_path" false; then
    echo "STOP: config gate failed: $config_path"
    exit 85
  fi
  output_dir="$(resolve_output_dir "$config_path")" || exit 86
  if [ -e "$output_dir" ]; then
    echo "STOP: output directory already exists: $output_dir"
    exit 87
  fi
done

for config_name in "${all_configs[@]:1}"; do
  config_path="configs/final/$config_name"
  if ! validate_config_gate "$config_path" true; then
    echo "STOP: CUDA gate failed: $config_path"
    exit 89
  fi
done

if [ "${CONFIG_GATE_ONLY:-0}" = '1' ]; then
  echo 'CONFIG_GATE=PASS count=19 trainable_cuda=18'
  exit 0
fi

nvidia-smi \
  --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
  --format=csv -l 5 \
  > results/final/gpu-monitor.csv 2>&1 &
gpu_monitor_pid=$!

if ! run_popular; then
  echo 'CAMPAIGN_STOP: Popular failed'
  exit 1
fi

gpu0_queue=(
  anime_bpr_final_seed42.yaml
  anime_bpr_final_seed44.yaml
  anime_bpr_ensemble_component_final_seed43.yaml
  anime_gmf_final_seed42.yaml
  anime_gmf_final_seed44.yaml
  anime_mlp_final_seed43.yaml
  anime_neumf_final_seed43.yaml
  anime_weighted_neumf_final_seed42.yaml
  anime_weighted_neumf_final_seed44.yaml
)

gpu1_queue=(
  anime_bpr_final_seed43.yaml
  anime_bpr_ensemble_component_final_seed42.yaml
  anime_bpr_ensemble_component_final_seed44.yaml
  anime_gmf_final_seed43.yaml
  anime_mlp_final_seed42.yaml
  anime_mlp_final_seed44.yaml
  anime_neumf_final_seed42.yaml
  anime_neumf_final_seed44.yaml
  anime_weighted_neumf_final_seed43.yaml
)

run_worker 0 "${gpu0_queue[@]}" > results/final/worker-gpu0.log 2>&1 &
worker0_pid=$!
run_worker 1 "${gpu1_queue[@]}" > results/final/worker-gpu1.log 2>&1 &
worker1_pid=$!

wait "$worker0_pid"
worker0_status=$?
wait "$worker1_pid"
worker1_status=$?

echo "WORKER_RESULT gpu=0 exit=$worker0_status"
echo "WORKER_RESULT gpu=1 exit=$worker1_status"

if [ "$worker0_status" -ne 0 ] || [ "$worker1_status" -ne 0 ]; then
  echo 'CAMPAIGN_STOP: at least one worker failed; no automatic retry'
  exit 1
fi

touch results/final/CAMPAIGN_COMPLETE
echo "CAMPAIGN_COMPLETE time=$(date --iso-8601=seconds)"
