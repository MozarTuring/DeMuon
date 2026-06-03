set -e

require_env() {
    for var in "$@"; do
        if [ -z "${!var}" ]; then
            echo "Error: $var is not set" >&2
            exit 1
        fi
    done
}

VENV_DIR=${RUN_DIR_PRE}/${RUN_PROJ}/venv
_REQ_FILE=requirements_gpt_cu128.txt
_REQ_HASH=$(md5sum "$_REQ_FILE" 2>/dev/null | awk '{print $1}')
_STAMP_FILE="${VENV_DIR}/.req_hash"

export PATH="$HOME/bin:$PATH"
MAMBA_ROOT="$HOME/micromamba"
MAMBA_ENV="$MAMBA_ROOT/envs/gcc_tools"
PYTHON_INCLUDE=$(find "$MAMBA_ENV/include" -name "Python.h" -printf '%h' -quit 2>/dev/null)
if [ -n "$PYTHON_INCLUDE" ]; then
    export CPATH="${PYTHON_INCLUDE}${CPATH:+:$CPATH}"
fi
echo "CC=$(which gcc) | CPATH=$CPATH"

if [ -d "$VENV_DIR" ] && [ -f "$_STAMP_FILE" ] && [ "$(cat "$_STAMP_FILE")" = "$_REQ_HASH" ]; then
    source "${VENV_DIR}/bin/activate"
else
    rm -rf "$VENV_DIR"
    python3 -m venv "$VENV_DIR"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install torch triton --index-url https://download.pytorch.org/whl/cu128
    pip install -r "$_REQ_FILE"
    echo "$_REQ_HASH" > "$_STAMP_FILE"
fi
JWM_RUN_COMMAND=(python launch.py)
"${JWM_RUN_COMMAND[@]}"  > job_out.log 2>&1 &
