# OCR PDF Folder Search

A PyQt6 desktop app that searches **scanned** PDFs. It uses
[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) to OCR the **front page**
of every PDF in a selected folder (recursively) and performs a case-insensitive
search for a term.

Because the PDFs are scanned images, there is no embedded text to extract — each
front page is rendered to an image and read with OCR.

## Features

- **Folder picker** (`File → Select Folder`) that recursively finds every PDF and
  lists its path (relative to the chosen folder) in the left panel.
- **Term search** (`Search → Term`) that OCRs the front page of each PDF and does a
  case-insensitive match.
- **Selectable concurrency** (`Concurrency` menu): a CPU process pool, or single-
  engine batching (which uses a CUDA GPU automatically if one is available).
- **Results panel** on the right lists every matching PDF; clicking a result (or
  any file on the left) shows its front page in the centre.
- **Remembers** the last selected folder and the last search term between runs.
- **PaddleOCR** engine (not Tesseract).
- **Persistent OCR cache**: OCR text is written to disk, so each PDF is only ever
  read once — interrupted searches, restarts, and new search terms all reuse the
  cached text (see below).

## Requirements

- [uv](https://docs.astral.sh/uv/) for dependency and environment management.
- Python 3.9–3.12 (uv installs the pinned version from `.python-version`
  automatically; PaddlePaddle does not yet support 3.13+).
- On first run, PaddleOCR downloads its detection/recognition models
  automatically (needs an internet connection once).

Dependencies (declared in `pyproject.toml`):

- `PyQt6` — desktop UI
- `paddleocr` + `paddlepaddle` — OCR engine (CPU build)
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
4. **`Concurrency`** — pick GPU batching or the CPU process pool before searching.
5. **`Search → Stop`** cancels an in-progress search.

The last folder and term are restored automatically the next time you launch the
app.

## How it works

1. **Discovery** — `os.walk` collects every `*.pdf` under the selected folder.
2. **Rendering** — PyMuPDF renders page 1 of each PDF to an image at
   `RENDER_DPI` (default 150) for both display and OCR.
3. **OCR** — PaddleOCR (`lang="en"`, angle classification on) reads the rendered
   page. Extracted text is cached to disk under `<folder>-text` (see
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

## Concurrency

Pick a strategy from the **`Concurrency`** menu before searching (the choice is
remembered between runs):

- **Process Pool (Executor)** — OCRs across multiple CPU worker processes, each
  pinned to a single math thread so they scale across cores. This is the default
  workhorse on CPU-only machines.
- **GPU Batching** — a single OCR engine fed by pages rendered ahead on a thread
  pool. It uses a CUDA GPU automatically when one is available; otherwise it runs
  on the CPU. The status bar reports which is in effect when a search starts.

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
- OCR language is set to English (`lang="en"`) in `SearchWorker.run`; change it
  there for other languages supported by PaddleOCR.

## Troubleshooting

- **First run is slow / needs network** — PaddleOCR is downloading its models.
- **`paddlepaddle` fails to install** — ensure Python is 3.9–3.12 (the project
  pins 3.12 via `.python-version`).
- **Blank / poor results** — increase `RENDER_DPI`, or verify the scan quality of
  the front page.
- **Out of GPU memory** — the app runs OCR on the CPU by default. If you added a
  GPU Paddle build, a small card can exhaust VRAM; revert to the CPU
  `paddlepaddle` build (see the GPU note under [Concurrency](#concurrency)).

## License

MIT — see `LICENSE`.
