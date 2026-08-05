# OCR PDF Folder Search

A PyQt6 desktop app that searches **scanned** PDFs. It OCRs the **front page**
of every PDF in a selected folder (recursively) and performs a case-insensitive
search for a term. The OCR engine is selectable from a menu —
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
[Tesseract](https://github.com/tesseract-ocr/tesseract), or
[RapidOCR](https://github.com/RapidAI/RapidOCR).

Because the PDFs are scanned images, there is no embedded text to extract — each
front page is rendered to an image and read with OCR.

## Features

- **Folder picker** (`File → Select Folder`) that recursively finds every PDF and
  lists its path (relative to the chosen folder) in the left panel.
- **Term search** (`Search → Term`) that OCRs the front page of each PDF and does a
  case-insensitive match.
- **Selectable OCR engine** (`OCR Engine` menu): PaddleOCR, Tesseract, or
  RapidOCR (see [OCR engine](#ocr-engine)).
- **Selectable concurrency** (`Concurrency` menu): a CPU process pool, or single-
  engine batching (which uses a CUDA GPU automatically if one is available).
- **Results panel** on the right lists every matching PDF; clicking a result (or
  any file on the left) shows its front page in the centre.
- **Remembers** the last selected folder, search term, engine, and concurrency
  choice between runs.
- **Persistent OCR cache**: OCR text is written to disk, so each PDF is only ever
  read once — interrupted searches, restarts, and new search terms all reuse the
  cached text (see below).

## Requirements

- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- Python 3.9–3.12 (uv installs the pinned version from `.python-version`
  automatically; PaddlePaddle does not yet support 3.13+).
- On first run, PaddleOCR and RapidOCR each download their detection/recognition
  models automatically (needs an internet connection once).
- **Tesseract only:** the external Tesseract binary must be installed separately
  and on `PATH` (see [OCR engine](#ocr-engine)). The other engines need no
  external binaries.

Dependencies (declared in `pyproject.toml`):

- `PyQt6` — desktop UI
- `paddleocr` + `paddlepaddle` — default OCR engine (CPU build)
- `rapidocr-onnxruntime` — RapidOCR engine (PP-OCR models on ONNXRuntime)
- `pytesseract` — Tesseract engine binding (needs the Tesseract binary)
- `PyMuPDF` — renders PDF pages to images (no external Poppler needed)
- `numpy`, `Pillow`

## Installation

```bash
uv sync
```

This creates a `.venv`, installs the pinned Python, and installs all
dependencies from the lock file.

## Usage

```bash
uv run main.py
```

Then:

1. **`File → Select Folder`** — choose the folder to search. All subfolders are
   included; every PDF appears in the left panel by relative path.
2. **`Search → Term`** — enter a term. The app OCRs the front pages concurrently
   (already-cached PDFs are matched instantly) and adds any matching PDFs to the
   results panel on the right, updating progress in the status bar.
3. Click any result (or any file on the left) to display its front page in the
   centre.
4. **`OCR Engine`** — pick PaddleOCR, Tesseract, or RapidOCR before searching.
5. **`Concurrency`** — pick GPU batching or the CPU process pool before searching.
6. **`Search → Stop`** cancels an in-progress search.

The last folder and term are restored automatically the next time you launch the
app.

## How it works

1. **Discovery** — `os.walk` collects every `*.pdf` under the selected folder.
2. **Rendering** — PyMuPDF renders page 1 of each PDF to an image at
   `RENDER_DPI` (default 100) for both display and OCR.
3. **OCR** — the selected engine (see [OCR engine](#ocr-engine)) reads the
   rendered page. Extracted text is cached to disk under `<folder>-text` (see
   [OCR text cache](#ocr-text-cache)) and reused on later searches.
4. **Search** — a case-insensitive substring match against the OCR text decides
   whether the PDF is a result.
5. **Concurrency** — OCR runs on a background `QThread` so the UI stays
   responsive. Uncached front pages are processed by the selected strategy —
   GPU batching (single engine + threaded rendering) or a CPU process pool —
   while cached PDFs are searched instantly on a cheap first pass. Progress and
   matches stream back to the main window.

## OCR text cache

When a search starts on folder `\a\b\c`, the app creates a sibling folder
`\a\b\c-text` that mirrors the original subfolder structure. The first-page OCR
text of each PDF is saved there as a `.txt` file:

```
\a\b\c\sub\d.pdf   ->   \a\b\c-text\sub\d.txt
```

Before OCRing a PDF, the app checks for its cached `.txt` file and, if present,
reuses it instead of running OCR again. This means:

- If a search is interrupted (or you hit `Search → Stop`), already-processed PDFs
  stay cached.
- Running a **different** search term on the same folder is near-instant, because
  no OCR is repeated — only the cached text is re-searched.

To force a fresh OCR pass, delete the `-text` folder (or the specific `.txt`
files) and search again.

## OCR engine

Pick the engine from the **`OCR Engine`** menu before searching (the choice is
remembered between runs). All three read the same rendered front page; they
differ in accuracy, memory behaviour, and setup:

- **PaddleOCR** (default) — the PP-OCR detection/recognition models on the
  PaddlePaddle runtime. Strong accuracy on noisy scans, but the PaddlePaddle
  runtime accumulates native memory across inference calls and cannot release it
  in-process, so on CPU it is always run through the recycling process pool (see
  the note below and [Concurrency](#concurrency)).
- **RapidOCR** — the **same PP-OCR models** run on ONNXRuntime instead of
  PaddlePaddle. Accuracy is essentially identical to PaddleOCR, but memory is
  stable and bounded and the install is much lighter. A good default if
  PaddleOCR's memory use is a problem.
- **Tesseract** — mature, very low and stable memory. Great on clean printed
  text, generally weaker than the PP-OCR models on low-quality scans. Requires
  the external Tesseract binary installed and on `PATH` (the `pytesseract`
  Python package alone is not enough). On Windows, install the
  [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki).

> **Memory note:** PaddleOCR's runtime is the reason RSS can climb to many GB on
> a folder of variably-sized scans — the leak is in the runtime, not the models.
> Switching to **RapidOCR** keeps the same models/accuracy while giving stable
> memory. See [Troubleshooting](#troubleshooting).

## Concurrency

Pick a strategy from the **`Concurrency`** menu before searching (the choice is
remembered between runs):

- **Process Pool (Executor)** — OCRs across multiple CPU worker processes, each
  pinned to a single math thread so they scale across cores. This is the default
  workhorse on CPU-only machines.
- **GPU Batching** — a single OCR engine fed by pages rendered ahead on a thread
  pool. It uses a CUDA GPU automatically when one is available; otherwise it runs
  the engine in-process on the CPU. The status bar reports the engine and the
  execution path in effect when a search starts.

The concurrency choice interacts with the [engine](#ocr-engine): because
PaddleOCR on CPU cannot release its memory in-process, selecting **GPU Batching**
with the **PaddleOCR** engine and no CUDA GPU automatically falls back to the
process pool (whose recycled workers reclaim memory by exiting). Tesseract and
RapidOCR have stable memory, so they keep the cheaper in-process batch path.

Either way, PDFs that already have cached OCR text are matched instantly on a
cheap first pass and never go through the concurrent pipeline.

> **GPU note:** the project ships with the **CPU** `paddlepaddle` build. Enabling
> a CUDA GPU means swapping in the `paddlepaddle-gpu` (CUDA) wheel plus a matching
> cuDNN, which pulls in large runtime libraries and needs enough VRAM for the OCR
> models — a small card (e.g. 6 GB) can run out of memory. It is left out here on
> purpose; add it back only if your GPU has headroom to spare.

## Configuration

- `RENDER_DPI` in `main.py` controls render/OCR resolution. Higher improves
  accuracy on low-quality scans but is slower and uses more memory.
- OCR language for PaddleOCR is set to English (`lang="en"`) in
  `_build_paddle_engine`; change it there for other supported languages.
- `EXECUTOR_PAGES_PER_WORKER` in `main.py` bounds how many pages each process-
  pool worker handles before the whole pool is torn down and replaced (chunk
  size is `DEFAULT_MAX_WORKERS × EXECUTOR_PAGES_PER_WORKER`). Lower values cap
  PaddleOCR's memory harder at the cost of more worker re-initialisation.
- The selected engine and concurrency strategy are remembered via `QSettings`.

## Troubleshooting

- **First run is slow / needs network** — PaddleOCR or RapidOCR is downloading
  its models.
- **`paddlepaddle` fails to install** — ensure Python is 3.9–3.12 (the project
  pins 3.12 via `.python-version`).
- **Memory climbs to many GB during a search** — this is the PaddlePaddle
  runtime accumulating native memory. Switch the **`OCR Engine`** menu to
  **RapidOCR** (same models/accuracy, stable memory), or lower
  `EXECUTOR_PAGES_PER_WORKER` to tear the pool down more often (see
  [OCR engine](#ocr-engine) and [Configuration](#configuration)).
- **Tesseract: `TesseractNotFoundError` / "OCR failed"** — the external
  Tesseract binary is not installed or not on `PATH`. Install it (on Windows, the
  [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki)); the
  `pytesseract` package alone is not enough.
- **Blank / poor results** — increase `RENDER_DPI`, try a different engine, or
  verify the scan quality of the front page.
- **Out of GPU memory** — the app runs OCR on the CPU by default. If you added a
  GPU Paddle build, a small card can exhaust VRAM; revert to the CPU
  `paddlepaddle` build (see the GPU note under [Concurrency](#concurrency)).

## License

MIT — see `LICENSE`.
