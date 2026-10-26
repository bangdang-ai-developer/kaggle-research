# kaggle-research

Workspace nghien cuu Kaggle: viet notebook local bang Claude Code, chay GPU tren Kaggle.

## Setup nhanh

```bash
uv sync          # tao venv + cai deps (Python 3.12 tu quan ly boi uv)
```

Credentials Kaggle: token nam o `~/.kaggle/access_token` (da co san). Kiem tra: `uv run kaggle config view`.

## Workflow

Xem [CLAUDE.md](CLAUDE.md) — vong lap: tao notebook tu `notebooks/_template/` → smoke-test local → `scripts/push_kernel.sh` → poll → `scripts/pull_output.sh` → ghi `notes/`.

## Cau truc

- `notebooks/` — trung tam, moi thu muc con = 1 kernel Kaggle
- `scripts/` — push / status / pull / fetch data
- `notes/` — log thi nghiem
- `data/`, `outputs/` — khong commit
