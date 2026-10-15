#!/usr/bin/env bash
# Sourced by run_ioher.sh. Mirrors import-heavy packages to node-local storage.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Source this file from run_ioher.sh; do not execute it directly." >&2
    exit 2
fi

: "${IOHER_IMPORT_STAGE_ROOT:?IOHER_IMPORT_STAGE_ROOT must be set}"

IOHER_IMPORT_PYTHON=${IOHER_IMPORT_PYTHON:-$(command -v python3)}
IOHER_IMPORT_STAGE_REFRESH=${IOHER_IMPORT_STAGE_REFRESH:-0}
IOHER_IMPORT_HOT_SITE="$IOHER_IMPORT_STAGE_ROOT/hot-site-packages"
IOHER_IMPORT_STAGE_LOCK="$IOHER_IMPORT_STAGE_ROOT/.stage.lock"

source_site=$(
    "$IOHER_IMPORT_PYTHON" -c \
        'import sysconfig; print(sysconfig.get_paths()["purelib"])'
)
if [[ ! -d "$source_site" ]]; then
    echo "Python site-packages directory does not exist: $source_site" >&2
    return 1
fi

mkdir -p "$IOHER_IMPORT_STAGE_ROOT"
exec 8>"$IOHER_IMPORT_STAGE_LOCK"
flock 8

# A package install changes its distribution metadata. This key makes repeated
# runs reuse the local mirror while rebuilding it after environment updates.
python_identity=$(
    "$IOHER_IMPORT_PYTHON" -c \
        'import platform, sys; print(sys.executable, platform.python_version())'
)
metadata_key=$(
    find "$source_site" -maxdepth 2 -type f \
        \( -path '*.dist-info/METADATA' -o -path '*.dist-info/RECORD' \) \
        -printf '%P:%s:%T@\n' | LC_ALL=C sort | sha256sum | awk '{print $1}'
)
stage_key=$(
    printf '%s\0%s\0%s\0%s\n' \
        1 "$source_site" "$python_identity" "$metadata_key" \
        | sha256sum | awk '{print $1}'
)
marker="$IOHER_IMPORT_HOT_SITE/.ioher-import-stage-key"

if [[ "$IOHER_IMPORT_STAGE_REFRESH" == 1 || ! -f "$marker" || "$(<"$marker")" != "$stage_key" ]]; then
    echo "IOHER_IMPORT_STAGE action=refresh source=$source_site"

    # The original environment remains on sys.path, so packages outside this
    # import-heavy set continue to resolve normally.
    patterns=(
        'accelerate*' 'aiohttp*' 'attrs*' 'certifi*' 'charset_normalizer*'
        'click*' 'cloudpickle*' 'codetiming*' 'cuda*' 'datasets*'
        'fastapi*' 'filelock*' 'flash_attn*' 'flashinfer*' 'fsspec*'
        'google' 'grpc*' 'huggingface_hub*' 'hydra*' 'idna*' 'jinja2*'
        'jsonschema*' 'latex2sympy2_extended*' 'llguidance*' 'lxml*'
        'markupsafe*' 'math_verify*' 'msgpack*' 'msgspec*' 'networkx*'
        'ninja*' 'numpy*' 'nvidia' 'omegaconf*' 'openai*' 'orjson*'
        'outlines*' 'packaging*' 'pandas*' 'peft*' 'prometheus_client*'
        'protobuf*' 'psutil*' 'pyarrow*' 'pydantic*' 'pyzmq*' 'ray*'
        'requests*' 'safetensors*' 'scipy*' 'setproctitle*' 'sgl_kernel*'
        'sglang*' 'starlette*' 'sympy*' 'tensordict*' 'tokenizers*'
        'torch*' 'transformers*' 'triton*' 'typing_extensions*' 'urllib3*'
        'uvicorn*' 'uvloop*' 'wandb*' 'xgrammar*' 'yaml' '_yaml*' 'zmq'
    )
    sources=()
    shopt -s nullglob
    for pattern in "${patterns[@]}"; do
        for candidate in "$source_site"/$pattern; do
            sources+=("$candidate")
        done
    done
    shopt -u nullglob
    if (( ${#sources[@]} == 0 )); then
        echo "No import packages found under $source_site" >&2
        return 1
    fi

    stage_tmp=$(mktemp -d "$IOHER_IMPORT_STAGE_ROOT/.hot-site-packages.XXXXXX")
    if ! rsync -a "${sources[@]}" "$stage_tmp/"; then
        rm -rf "$stage_tmp"
        return 1
    fi
    printf '%s\n' "$stage_key" > "$stage_tmp/.ioher-import-stage-key"
    rm -rf "$IOHER_IMPORT_HOT_SITE"
    mv "$stage_tmp" "$IOHER_IMPORT_HOT_SITE"
else
    echo "IOHER_IMPORT_STAGE action=reuse path=$IOHER_IMPORT_HOT_SITE"
fi

for required in torch transformers ray sglang; do
    if [[ ! -e "$IOHER_IMPORT_HOT_SITE/$required" ]]; then
        echo "Incomplete local import stage: missing $required" >&2
        return 1
    fi
done

local_library_paths=("$IOHER_IMPORT_HOT_SITE/torch/lib")
while IFS= read -r library_dir; do
    local_library_paths+=("$library_dir")
done < <(find "$IOHER_IMPORT_HOT_SITE/nvidia" -mindepth 2 -maxdepth 2 \
    -type d -name lib 2>/dev/null | LC_ALL=C sort)
local_library_path=$(IFS=:; echo "${local_library_paths[*]}")

export IOHER_IMPORT_HOT_SITE
export PYTHONPATH="$IOHER_IMPORT_HOT_SITE:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="$local_library_path:${LD_LIBRARY_PATH:-}"

flock -u 8
exec 8>&-
echo "IOHER_IMPORT_STAGE action=ready path=$IOHER_IMPORT_HOT_SITE"
