HALAL JORDAN — Read Me First
============================

This is the laptop-safe / thumbdrive build of Halal Jordan, a local,
offline, source-grounded Islamic research assistant.

Your first 60 seconds:

  1. Double-click  START HALAL JORDAN.cmd
  2. Wait ~10 seconds (the preflight check runs, then the server starts).
  3. Your browser will open at http://127.0.0.1:8000/
     (If it doesn't, open that URL yourself.)
  4. Type a question, click Ask. That's it.

What you'll see at /:

  - Question box. One field, one button.
  - Profile chip. Click it to switch reasoning modes
    (Shaykh Jamal, Dr. Umar, Hadith-Focused, Hanafi-Heavy, etc).
  - Answer area. Sources, Evidence Ladder, Confidence label.

To stop:

  - Double-click  STOP_HALAL_JORDAN.ps1   (right-click > Run with PowerShell)
  - OR just close the terminal window the launcher opened.


Running from a USB thumbdrive
-----------------------------

This build is designed for thumbdrive use:

  - All code, all sources, the embedding model, and the Python runtime
    are bundled in this folder. No internet required at runtime.
  - Works from any drive letter (D:, E:, F:, etc).
  - First boot takes ~10s; subsequent boots take ~5s.

Requirements for the thumbdrive:

  - exFAT or NTFS filesystem (NOT FAT32 — file size limits will break it).
  - At least 8 GB free.
  - USB 3.0 strongly recommended (USB 2.0 will work but boots more slowly).

If Windows Defender or your antivirus blocks the launcher:

  - This is expected for unsigned scripts. The bundled PowerShell
    launcher is safe, but until we code-sign it you may need to
    right-click START_HALAL_JORDAN.ps1 > Properties > Unblock,
    or run it from PowerShell explicitly:
        powershell.exe -ExecutionPolicy Bypass -File START_HALAL_JORDAN.ps1


For developers / advanced users
-------------------------------

  - LAUNCH_HALAL_JORDAN.ps1   — full launcher with all options
                                 (-Port N, -NoBrowser, -ServerOnly, etc).
  - HJ_PORT environment var   — override the default port 8000.
                                 If busy, launcher tries 8001-8009 automatically.
  - /workspace                — the full chat / projects / memory UI
                                 (for users who want LLM synthesis;
                                 requires Ollama for chat responses).
  - /profiles                 — change the active reasoning profile.
  - /admin                    — runtime config, logs, diagnostics.

For the project mission, charter, and reasoning architecture, see
VISION.md and TONE.md.

For first-run setup details and troubleshooting, see FIRST_RUN.md.
