#!/usr/bin/env sh
# TerraceRoute Chat entrypoint. Default: launch the Streamlit chatbot.
# `docker run <img>`        -> chatbot UI
# `docker run <img> chat`   -> chatbot UI (explicit)
# `docker run <img> <cmd>`  -> run <cmd> (e.g. `sh` for debugging)
set -e

if [ -z "$1" ] || [ "$1" = "chat" ]; then
    exec streamlit run /app/chat_app.py
fi
exec "$@"
