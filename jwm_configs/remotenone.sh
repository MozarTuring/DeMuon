# JWM_SERVER_NAME=greatrawr

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
