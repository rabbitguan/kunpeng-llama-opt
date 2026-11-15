#!/usr/bin/env bash
set -euo pipefail

# Keep dependency paths fixed; venv and build files can contain absolute paths.
mode=${1:---list}
case "$mode" in
    --list|--apply) ;;
    *) echo 'Usage: bash cleanup_server.sh [--list|--apply]' >&2; exit 2 ;;
esac
base=/data/home/2300013234
project=$base/kunpeng-llama-opt
[[ -d "$project/tmac" && -d "$base/kunpeng-llm/models" ]] || {
    echo 'Expected server environment and model directories are missing.' >&2
    exit 1
}
paths=(
    "$project/tmac_compare"
    "$project/q4_final_experiment"
    "$project/q4_stage_experiment"
    "$project/q4_kernel_lab"
    "$project/tmac_baseline"
    "$project/tmac_experiment"
    "$project/results"
    "$project/.pkl_memoize_py3"
    "$project/tmac_baseline_cnkunpengb_v1 .zip"
    "$project/tmac_compare_cnkunpengb_v2.zip"
    "$base/kunpeng-llm/results"
    "$base/kunpeng-llm/logs"
    "$base/kunpeng-llm/old_experiment_inventory.txt"
    "$base/kunpeng-llm/old_experiment_key_details.txt"
    "$base/cpu_supplement_results"
    "$base/cpu_cnkunpengb_cnkunpengb01_20261027T113042Z_7d4b1be0"
)
shopt -s nullglob
paths+=("$project"/tmac_experiment.bak.*)
existing=()
for path in "${paths[@]}"; do
    [[ -e "$path" || -L "$path" ]] || continue
    [[ ! -L "$path" ]] || { echo "Refusing symlink: $path" >&2; exit 1; }
    if [[ -d "$path" ]]; then
        marker=$(find "$path" -type f \( -name pyvenv.cfg -o -name 'libLLVM*' -o -name 'libtvm*' -o -path '*/bin/activate' \) -print -quit)
        [[ -z "$marker" ]] || { echo "Environment file found; nothing deleted: $marker" >&2; exit 1; }
    fi
    existing+=("$path")
done
printf 'Selected old experiment files and directories:\n'
if ((${#existing[@]})); then
    printf '%s\n' "${existing[@]}"
    if [[ "$mode" == --apply ]]; then
        rm -rf -- "${existing[@]}"
        echo 'Cleanup completed.'
    fi
fi
printf 'Preserved: repo, tmac, setup, builds, tmac_baseline_clean_source, models, tools and account configuration.\n'
