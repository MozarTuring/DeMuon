# JWM_SERVER_NAME=

VENV_DIR=${RUN_DIR_PRE}/${RUN_PROJ}/venv
_REQ_FILE=requirements_gpt_cu128.txt
_REQ_HASH=$(md5sum "$_REQ_FILE" 2>/dev/null | awk '{print $1}')
_STAMP_FILE="${VENV_DIR}/.req_hash"

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
