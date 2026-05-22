# Halal Jordan — First Run Guide

This guide is for the person plugging in the thumbdrive for the
first time. If you're a developer, see [VISION.md](VISION.md) and
the source tree.

---

## What to expect

| Step | Time | What happens |
|---|---|---|
| 1. Double-click launcher | < 1s | A terminal window opens. |
| 2. Preflight check | < 1s | Verifies bundled Python, site-packages, embedding model, and index are present. Prints `[OK]` for each. |
| 3. Python launch | 2–3s | The bundled Python interpreter starts and loads dependencies. |
| 4. Server start | 1–2s | FastAPI binds to port 8000 (or the next available port). |
| 5. Browser opens | < 1s | Your default browser opens at the active URL. |
| 6. First question | 1–4s | Retrieval runs. The persisted index is read from disk; no rebuild needed. |

Total cold start: **5–10 seconds**.

---

## What if it doesn't work

### Windows Defender blocks the launcher

This is the most common first-time issue. The bundled PowerShell
script is unsigned (code-signing requires a paid certificate).

**Fix:**

1. Right-click `START_HALAL_JORDAN.ps1` → **Properties** → tick
   **Unblock** → **Apply** → **OK**.
2. If that doesn't work, open a PowerShell terminal in this folder
   and run:
   ```
   powershell.exe -ExecutionPolicy Bypass -File START_HALAL_JORDAN.ps1
   ```

### "Port 8000 is in use" — but the launcher tried 8000–8009

That means you have at least 10 other apps using local ports in that
range. Set a different port and retry:

```
$env:HJ_PORT = "8500"
.\START_HALAL_JORDAN.ps1
```

### Browser opens but the page doesn't load

The server may still be warming up. Wait 10 seconds and refresh.

If after 30 seconds the page still doesn't load:

- Open `logs\halal-jordan-launch.log` and look for errors.
- Check `logs\halal-jordan-app.err.log` for Python tracebacks.

### Preflight says "MISSING: Bundled Python interpreter"

The bundle is incomplete. You likely copied the project folder but
missed the `runtime/` subdirectory.

- The `runtime/` folder contains the Python interpreter, all
  Python packages, the embedding model, and the llama runtime.
- Total size of `runtime/` is ~1.7 GB.
- Re-copy the entire project from the source, including `runtime/`,
  `models/`, `data/`, and `app/`.

### Preflight says "WARN: Bundled embedding model"

The system will still work, but semantic search falls back to
keyword-only. Some questions that would have matched related
language won't.

- The embedding model lives at `runtime/huggingface/models--sentence-transformers--all-MiniLM-L6-v2/`.
- Total size ~88 MB.
- Re-copy that subdirectory from the source.

### "FAT32" — file size errors

FAT32 thumbdrives have a 4 GB per-file limit. Several files in this
build (model weights, the Python interpreter bundle, large PDFs) may
exceed that.

**Fix:** reformat the thumbdrive as **exFAT** (preserves cross-OS
compatibility) or **NTFS** (Windows-only but no size limits).

---

## How to know it's working

After the browser opens at `http://127.0.0.1:8000/`:

1. Type a question — for example: `What do hadith say about intentions?`
2. Click **Ask**.
3. Within 1–4 seconds you should see:
   - A list of cited sources.
   - An **Evidence Ladder** section showing the sources grouped by
     epistemological tier (Qur'an, Sahih Hadith, etc).
   - A **Scholarly Confidence** label (e.g. "Explicit Text" or
     "Valid Disagreement").
4. Click the profile chip at the top right. You should see a page
   with big buttons for each reasoning profile.

If all three of those work, the system is fully operational.

---

## Where things live

| Folder | What's in it |
|---|---|
| `app/` | The Python application code. |
| `config/` | Runtime configuration (which profile is active, etc). |
| `data/raw/` | Source PDFs and text (Quran, hadith collections, fiqh, tasawwuf). |
| `data/index/` | The persisted retrieval index. |
| `data/processed/` | Normalized text extracted from PDFs. |
| `models/` | LLM weights (used by `/workspace` chat synthesis). |
| `runtime/` | Bundled Python interpreter, packages, llama runtime, HF cache. |
| `logs/` | Logs from launcher and server runs. |
| `metadata/` | Schemas, taxonomies, scholar profiles. |
| `prompts/` | LLM prompt templates. |

---

## Charter reading

If you want to understand what this system is and isn't:

- [VISION.md](VISION.md) — the project charter. What Halal Jordan
  *is* (a transparent, source-grounded research assistant) and what
  it is **not** (a fatwa engine, an AI mufti, a replacement scholar).
- [TONE.md](TONE.md) — how the system speaks. The five emotional
  states it recognizes; the eight non-negotiable tone rules.

---

## Stopping the system

Three ways:

1. Close the terminal window the launcher opened.
2. Right-click `STOP_HALAL_JORDAN.ps1` → **Run with PowerShell**.
3. From PowerShell in this folder: `.\STOP_HALAL_JORDAN.ps1`.

The thumbdrive is then safe to eject.
