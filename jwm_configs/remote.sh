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
if [ -d "$VENV_DIR" ]; then
    source "${VENV_DIR}/bin/activate"
else
    python3 -m venv "$VENV_DIR"
    source "${VENV_DIR}/bin/activate"
    pip install --upgrade pip
    pip install -r requirements_gpt.txt
fi
JWM_RUN_COMMAND=(python launch.py)
