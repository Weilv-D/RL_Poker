# RL Poker Playground

This directory contains interactive interfaces for playing against the RL AI.

## Requirements

- Install the project in editable mode: `pip install -e .`
- Playground extras (not in core deps): `pip install fastapi uvicorn pydantic rich`

## 1. Terminal UI (TUI)
Enhanced terminal interface with big cards.

```bash
# From project root
python playground/play_tui.py --tui
```

## 2. Web Interface
Modern web-based GUI (FastAPI + static frontend).

```bash
# From project root
python playground/web_server.py
```

Open `http://127.0.0.1:8000` in your browser. Optional env vars:

- `RL_POKER_HOST` / `RL_POKER_PORT` to bind a different address/port
- `RL_POKER_API_TOKEN` to require a token for `/api/*` (open `/?token=...` once)

If no checkpoints are found under `checkpoints/`, the UI falls back to random opponents.
