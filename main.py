"""PyQt6 desktop app to OCR the front page of scanned PDFs and search a term.

The app lets the user pick a folder (recursively scanned for PDFs), then runs
PaddleOCR on the front page of every PDF and performs a case-insensitive search
for a term. OCR of uncached PDFs runs concurrently using a selectable strategy
(see the Concurrency menu); files that already have cached OCR text are handled
cheaply up front and never go through the concurrent pipeline. Matches are
collected on the right. Selecting a file or a result renders its front page in
the centre.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool

import fitz  # PyMuPDF
import numpy as np
from PyQt6.QtCore import QObject, QSettings, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QAction, QActionGroup, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Higher DPI improves OCR accuracy on scanned pages at the cost of speed/memory.
RENDER_DPI = 150

# Concurrency strategy identifiers (persisted in QSettings and shown in the menu).
CONCURRENCY_GPU_BATCH = "gpu_batch"
CONCURRENCY_EXECUTOR = "executor"
DEFAULT_CONCURRENCY = CONCURRENCY_GPU_BATCH

# OCR engine identifiers (persisted in QSettings and shown in the menu). The
# engine is the backend that actually reads text; the concurrency setting only
# decides how work is spread across threads/processes.
ENGINE_PADDLE = "paddle"
ENGINE_TESSERACT = "tesseract"
ENGINE_RAPIDOCR = "rapidocr"
DEFAULT_ENGINE = ENGINE_PADDLE
VALID_ENGINES = (ENGINE_PADDLE, ENGINE_TESSERACT, ENGINE_RAPIDOCR)

# GPU-batching pipeline: how many front pages to pre-render per round, and how
# many threads to render them with (PyMuPDF rendering releases the GIL).
OCR_BATCH_SIZE = 30
RENDER_WORKERS = 4

# Process-pool executor: default number of worker processes. Each worker is
# pinned to a single math thread (see ``_pin_cpu_threads``) so that N processes
# scale across cores instead of oversubscribing them, which lets us run more of
# them. Capped to keep memory (one PaddleOCR engine per process) in check.
DEFAULT_MAX_WORKERS = min(8, os.cpu_count() or 2)

# PaddleOCR/Paddle accumulate native memory across inference calls and cannot
# return it to the OS without exiting the process. ``ProcessPoolExecutor``'s
# ``max_tasks_per_child`` is unreliable on Windows (the pool often breaks after
# the first generation of workers, aborting the rest of the search at exactly
# ``workers × pages`` — e.g. 8×4 = 32). Instead we process files in chunks of
# ``DEFAULT_MAX_WORKERS × EXECUTOR_PAGES_PER_WORKER``, fully shutting the pool
# down between chunks so memory is reclaimed and the next chunk starts fresh.
EXECUTOR_PAGES_PER_WORKER = 4

# PaddleOCR is imported lazily inside the worker thread/processes because
# importing it (and initialising the models) is slow and prints a lot of noise.


def render_front_page(path: str, dpi: int = RENDER_DPI) -> tuple[QImage, np.ndarray]:
    """Render page 1 of a PDF to a QImage (for display) and an RGB ndarray (for OCR)."""
    doc = fitz.open(path)
    try:
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        image = QImage(
            pix.samples,
            pix.width,
            pix.height,
            pix.stride,
            QImage.Format.Format_RGB888,
        ).copy()
        array = (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, pix.n)
            .copy()
        )
        return image, array
    finally:
        doc.close()


def render_front_page_array(path: str, dpi: int = RENDER_DPI) -> np.ndarray:
    """Render page 1 of a PDF to an RGB ndarray only (no Qt dependency).

    Used by the concurrent OCR strategies, including inside worker processes
    where importing/instantiating Qt objects is neither needed nor desirable.
    """
    doc = fitz.open(path)
    try:
        page = doc.load_page(0)
        pix = page.get_pixmap(dpi=dpi, alpha=False)
        return (
            np.frombuffer(pix.samples, dtype=np.uint8)
            .reshape(pix.height, pix.width, pix.n)
            .copy()
        )
    finally:
        doc.close()


def text_dir_for(folder: str) -> str:
    """Sibling directory that mirrors ``folder`` and holds cached OCR text.

    Example: ``\\a\\b\\c`` -> ``\\a\\b\\c-text``.
    """
    return os.path.normpath(folder) + "-text"


def text_path_for(pdf_path: str, folder: str, text_dir: str) -> str:
    """Path of the ``.txt`` cache file mirroring ``pdf_path`` under ``text_dir``.

    Example: ``\\a\\b\\c\\sub\\d.pdf`` -> ``\\a\\b\\c-text\\sub\\d.txt``.
    """
    rel = os.path.relpath(pdf_path, folder)
    return os.path.join(text_dir, os.path.splitext(rel)[0] + ".txt")


def texts_from_paddle_result(result) -> list[str]:
    """Flatten a PaddleOCR (2.7.x) result into a list of detected text strings."""
    texts: list[str] = []
    if not result:
        return texts
    for page in result:
        if not page:
            continue
        for line in page:
            try:
                texts.append(line[1][0])
            except (IndexError, TypeError):
                continue
    return texts


def gpu_available() -> bool:
    """True if the installed PaddlePaddle build can actually use a CUDA GPU."""
    try:
        import paddle

        return bool(paddle.is_compiled_with_cuda())
    except Exception:  # noqa: BLE001 - any import/probe failure means "no GPU"
        return False


def _pin_cpu_threads(threads: int = 1) -> None:
    """Constrain native math-library threads for this process.

    Set before PaddlePaddle is imported so its OpenMP/MKL backend honours it.
    With one math thread per worker, a process pool scales across cores instead
    of each worker oversubscribing them (the reason CPU parallelism looked flat).
    """
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(var, str(threads))


def _build_paddle_engine(use_gpu: bool, cpu_threads: int | None = None):
    """Instantiate a PaddleOCR engine, tolerating minor API differences.

    Falls back to CPU if a GPU engine cannot be created, so a strategy can
    request the GPU without hard-failing.
    """
    from paddleocr import PaddleOCR

    base = {"use_angle_cls": True, "lang": "en"}
    thread_opts = {} if cpu_threads is None else {"cpu_threads": cpu_threads}

    # Ordered most-preferred -> minimal, degrading kwargs the API may reject.
    attempts: list[dict] = []
    if use_gpu:
        attempts.append({**base, **thread_opts, "show_log": False, "use_gpu": True})
    attempts.append({**base, **thread_opts, "show_log": False})
    attempts.append({**base, "show_log": False})
    attempts.append({**base})

    last_exc: Exception | None = None
    for kwargs in attempts:
        try:
            return PaddleOCR(**kwargs)
        except Exception as exc:  # noqa: BLE001 - try the next, less specific, signature
            last_exc = exc
    raise RuntimeError(f"Could not initialise PaddleOCR: {last_exc}")


# --------------------------------------------------------------- OCR backends
# Each backend wraps one OCR engine behind a uniform ``ocr_text(array) -> str``
# call so the concurrency strategies don't care which engine is in use. Heavy
# third-party imports are done lazily inside ``__init__`` so a missing optional
# dependency only fails when that engine is actually selected.


class PaddleBackend:
    """PaddleOCR (PP-OCR models on the PaddlePaddle runtime)."""

    def __init__(self, use_gpu: bool = False, cpu_threads: int | None = None):
        self._engine = _build_paddle_engine(use_gpu=use_gpu, cpu_threads=cpu_threads)

    def ocr_text(self, array: np.ndarray) -> str:
        result = self._engine.ocr(array, cls=True)
        return "\n".join(texts_from_paddle_result(result))


class TesseractBackend:
    """Tesseract via ``pytesseract`` (needs the Tesseract binary installed)."""

    def __init__(self, use_gpu: bool = False, cpu_threads: int | None = None):
        import pytesseract  # noqa: F401 - probe availability early

        self._pytesseract = pytesseract

    def ocr_text(self, array: np.ndarray) -> str:
        from PIL import Image

        return self._pytesseract.image_to_string(Image.fromarray(array))


class RapidOcrBackend:
    """RapidOCR: the PP-OCR models run on ONNXRuntime (stable, bounded memory)."""

    def __init__(self, use_gpu: bool = False, cpu_threads: int | None = None):
        from rapidocr_onnxruntime import RapidOCR

        self._engine = RapidOCR()

    def ocr_text(self, array: np.ndarray) -> str:
        result, _elapse = self._engine(array)
        if not result:
            return ""
        # RapidOCR yields [box, text, score] triples; keep the text field.
        return "\n".join(line[1] for line in result if len(line) > 1)


def build_backend(
    engine: str, use_gpu: bool = False, cpu_threads: int | None = None
):
    """Instantiate the OCR backend for the given engine identifier."""
    if engine == ENGINE_TESSERACT:
        return TesseractBackend(use_gpu=use_gpu, cpu_threads=cpu_threads)
    if engine == ENGINE_RAPIDOCR:
        return RapidOcrBackend(use_gpu=use_gpu, cpu_threads=cpu_threads)
    return PaddleBackend(use_gpu=use_gpu, cpu_threads=cpu_threads)


def engine_uses_gpu(engine: str) -> bool:
    """True only when the selected engine can actually use a CUDA GPU here.

    Only the PaddlePaddle build is probed for CUDA; the Tesseract and RapidOCR
    backends shipped here run on the CPU.
    """
    return engine == ENGINE_PADDLE and gpu_available()


def _safe_render(path: str) -> tuple[np.ndarray | None, str | None]:
    """Render a front page for the batching pipeline, capturing any error."""
    try:
        return render_front_page_array(path), None
    except Exception as exc:  # noqa: BLE001 - reported per-file, keep going
        return None, str(exc)


# Per-process OCR backend, lazily created inside each executor worker, plus the
# engine id it was built for (so a worker recycled without the initializer can
# rebuild the right backend).
_WORKER_BACKEND = None
_WORKER_ENGINE_ID = DEFAULT_ENGINE


def _executor_init(engine: str) -> None:
    """Worker initializer: record the engine and pin math threads.

    The backend itself is built lazily on the first task (see ``_executor_ocr``)
    rather than here, so a backend that fails to construct surfaces as a per-file
    error instead of an initializer exception that could stall the pool.
    """
    global _WORKER_ENGINE_ID
    _WORKER_ENGINE_ID = engine
    _pin_cpu_threads(1)


def _executor_ocr(path: str) -> tuple[str, str | None, str | None]:
    """Render + OCR one PDF inside a worker process. Returns (path, text, error)."""
    global _WORKER_BACKEND
    try:
        if _WORKER_BACKEND is None:  # first task in this (possibly recycled) worker
            _pin_cpu_threads(1)
            _WORKER_BACKEND = build_backend(
                _WORKER_ENGINE_ID, use_gpu=False, cpu_threads=1
            )
        array = render_front_page_array(path)
        return path, _WORKER_BACKEND.ocr_text(array), None
    except Exception as exc:  # noqa: BLE001 - surfaced as a per-file failure
        return path, None, str(exc)


# Cached batching backend, reused across searches. Building an OCR engine is
# expensive, so keep a single instance keyed by (engine, gpu) and rebuild it only
# when that key changes. (PaddleOCR on CPU is routed to the process pool instead
# of this path precisely because its memory cannot be reclaimed in-process.)
_BATCH_BACKEND = None
_BATCH_BACKEND_KEY: tuple[str, bool] | None = None


def _get_batch_backend(engine: str, use_gpu: bool):
    """Return a shared OCR backend, building it once per (engine, gpu) key."""
    global _BATCH_BACKEND, _BATCH_BACKEND_KEY
    key = (engine, use_gpu)
    if _BATCH_BACKEND is None or _BATCH_BACKEND_KEY != key:
        _BATCH_BACKEND = build_backend(engine, use_gpu=use_gpu)
        _BATCH_BACKEND_KEY = key
    return _BATCH_BACKEND


class GpuBatchStrategy:
    """Single engine fed by a batch of pre-rendered pages.

    The speed-up here comes from overlapping page rendering across threads while
    the engine works (and, for a CUDA-capable PaddlePaddle build, running
    detection/recognition on the GPU). Any of the OCR backends can be used.
    """

    def __init__(
        self,
        engine: str = DEFAULT_ENGINE,
        batch_size: int = OCR_BATCH_SIZE,
        render_workers: int = RENDER_WORKERS,
    ):
        self._engine = engine
        self._batch_size = batch_size
        self._render_workers = render_workers

    def process(self, files, stop_check):
        use_gpu = engine_uses_gpu(self._engine)
        with ThreadPoolExecutor(max_workers=self._render_workers) as renderer:
            for start in range(0, len(files), self._batch_size):
                if stop_check():
                    return
                batch = files[start : start + self._batch_size]
                rendered = list(renderer.map(_safe_render, batch))
                for path, (array, err) in zip(batch, rendered):
                    if stop_check():
                        return
                    if err is not None:
                        yield path, None, err
                        continue
                    backend = _get_batch_backend(self._engine, use_gpu)
                    try:
                        text = backend.ocr_text(array)
                    except Exception as exc:  # noqa: BLE001 - per-file failure
                        yield path, None, str(exc)
                        continue
                    finally:
                        # Drop the rendered page as soon as OCR is done with it.
                        del array
                    yield path, text, None
                del rendered


class ExecutorStrategy:
    """OCR across a pool of worker processes, each with its own engine."""

    def __init__(self, engine: str = DEFAULT_ENGINE, max_workers: int = DEFAULT_MAX_WORKERS):
        self._engine = engine
        self._max_workers = max(1, max_workers)

    def process(self, files, stop_check):
        # Process in finite chunks, tearing the pool down between them. That
        # reclaims PaddleOCR's leaked native memory without relying on
        # ``max_tasks_per_child``, which on Windows often breaks the pool after
        # the first generation of workers and silently aborts the rest.
        chunk_size = self._max_workers * EXECUTOR_PAGES_PER_WORKER
        for start in range(0, len(files), chunk_size):
            if stop_check():
                return
            chunk = files[start : start + chunk_size]
            pool = ProcessPoolExecutor(
                max_workers=self._max_workers,
                initializer=_executor_init,
                initargs=(self._engine,),
            )
            try:
                futures = {pool.submit(_executor_ocr, path): path for path in chunk}
                for future in as_completed(futures):
                    if stop_check():
                        return
                    try:
                        path, text, err = future.result()
                    except BrokenProcessPool:
                        # A broken pool is unrecoverable (e.g. engine init crashed),
                        # so let the worker treat it as a fatal error.
                        raise
                    except Exception as exc:  # noqa: BLE001 - per-file failure
                        yield futures[future], None, str(exc)
                        continue
                    yield path, text, err
            finally:
                # Wait for workers to exit so the OS reclaims each worker's OCR
                # engine before the next chunk spins up a fresh pool.
                pool.shutdown(wait=True, cancel_futures=True)


def make_strategy(concurrency: str, engine: str = DEFAULT_ENGINE):
    """Return an OCR strategy for the given concurrency + engine identifiers.

    PaddleOCR on CPU cannot release its process-global native memory (rebuilding
    the engine in-place does not return the allocator/oneDNN arenas to the OS, so
    RSS only grows), so single-engine batching is forced onto the process pool,
    whose recycled workers actually reclaim memory by exiting. Tesseract and
    RapidOCR have stable memory, so their in-process batch path is kept.
    """
    if concurrency == CONCURRENCY_EXECUTOR:
        return ExecutorStrategy(engine)
    if engine == ENGINE_PADDLE and not gpu_available():
        return ExecutorStrategy(engine)
    return GpuBatchStrategy(engine)


class SearchWorker(QObject):
    """Runs OCR + search on a background thread and streams progress to the UI."""

    match_found = pyqtSignal(str)  # matched path
    progress = pyqtSignal(int, int)  # done, total
    failed = pyqtSignal(str, str)  # path, error message
    finished = pyqtSignal(int)  # number of matches
    fatal = pyqtSignal(str)  # unrecoverable error (e.g. engine init)
    status = pyqtSignal(str)  # transient status note (e.g. GPU vs CPU)

    def __init__(
        self,
        files: list[str],
        term: str,
        folder: str,
        text_dir: str,
        text_cache: dict[str, str],
        concurrency: str = DEFAULT_CONCURRENCY,
        engine: str = DEFAULT_ENGINE,
    ):
        super().__init__()
        self._files = files
        self._term = term.lower()
        self._folder = folder
        self._text_dir = text_dir
        self._text_cache = text_cache
        self._concurrency = concurrency
        self._engine = engine
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def _cached_text(self, path: str) -> str | None:
        """Return OCR text from the in-memory or on-disk cache, or None."""
        if path in self._text_cache:
            return self._text_cache[path]
        txt_path = text_path_for(path, self._folder, self._text_dir)
        if os.path.isfile(txt_path):
            try:
                with open(txt_path, encoding="utf-8") as handle:
                    text = handle.read()
                self._text_cache[path] = text
                return text
            except OSError:
                return None
        return None

    def _store_text(self, path: str, text: str) -> None:
        """Persist OCR text to the mirrored ``<folder>-text`` tree and memory."""
        self._text_cache[path] = text
        txt_path = text_path_for(path, self._folder, self._text_dir)
        try:
            os.makedirs(os.path.dirname(txt_path), exist_ok=True)
            with open(txt_path, "w", encoding="utf-8") as handle:
                handle.write(text)
        except OSError as exc:  # non-fatal: caching is best-effort
            self.failed.emit(path, f"Could not cache OCR text: {exc}")

    def _strategy_status(self) -> str:
        """Human-readable note about which engine + execution path will run."""
        engine_names = {
            ENGINE_PADDLE: "PaddleOCR",
            ENGINE_TESSERACT: "Tesseract",
            ENGINE_RAPIDOCR: "RapidOCR",
        }
        engine_name = engine_names.get(self._engine, self._engine)
        if self._engine == ENGINE_PADDLE and gpu_available():
            return f"{engine_name} \u2014 GPU batching on CUDA"
        strategy = make_strategy(self._concurrency, self._engine)
        if isinstance(strategy, ExecutorStrategy):
            return f"{engine_name} \u2014 process pool ({DEFAULT_MAX_WORKERS} workers)"
        return f"{engine_name} \u2014 in-process batching"

    def run(self) -> None:
        total = len(self._files)
        done = 0
        matches = 0

        # First pass: files with cached OCR text need no rendering or OCR, so
        # handle them immediately and keep them out of the concurrent pipeline.
        uncached: list[str] = []
        for path in self._files:
            if self._stop:
                break
            text = self._cached_text(path)
            if text is None:
                uncached.append(path)
                continue
            done += 1
            if self._term in text.lower():
                matches += 1
                self.match_found.emit(path)
            self.progress.emit(done, total)

        # Second pass: OCR the remaining files with the chosen concurrency
        # strategy, storing/searching each result as it streams back.
        if not self._stop and uncached:
            self.status.emit(self._strategy_status())
            strategy = make_strategy(self._concurrency, self._engine)
            try:
                for path, text, err in strategy.process(uncached, lambda: self._stop):
                    if err is not None:
                        self.failed.emit(path, err)
                    else:
                        self._store_text(path, text)
                        if self._term in text.lower():
                            matches += 1
                            self.match_found.emit(path)
                    done += 1
                    self.progress.emit(done, total)
                    if self._stop:
                        break
            except Exception as exc:  # noqa: BLE001 - engine init / broken pool
                self.fatal.emit(f"OCR failed: {exc}")
                return

        self.finished.emit(matches)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OCR PDF Folder Search")
        self.resize(1280, 800)

        self.settings = QSettings("OcrPdfFolderSearch", "OcrPdfFolderSearch")
        self.folder: str | None = None
        self.pdf_paths: list[str] = []
        self.text_cache: dict[str, str] = {}
        self.thread: QThread | None = None
        self.worker: SearchWorker | None = None
        # Guards against re-rendering when a result selection syncs the file list.
        self._syncing_selection = False
        self.concurrency = self.settings.value(
            "concurrency", DEFAULT_CONCURRENCY, type=str
        )
        if self.concurrency not in (CONCURRENCY_GPU_BATCH, CONCURRENCY_EXECUTOR):
            self.concurrency = DEFAULT_CONCURRENCY
        self.engine = self.settings.value("engine", DEFAULT_ENGINE, type=str)
        if self.engine not in VALID_ENGINES:
            self.engine = DEFAULT_ENGINE

        self._build_ui()
        self._build_menu()

        last_folder = self.settings.value("lastFolder", "", type=str)
        if last_folder and os.path.isdir(last_folder):
            self.load_folder(last_folder)

    # ---------------------------------------------------------------- UI setup
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: all PDFs (relative paths).
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("PDFs"))
        self.file_list = QListWidget()
        self.file_list.currentItemChanged.connect(self._on_file_selected)
        left_layout.addWidget(self.file_list)

        # Centre: front-page image display.
        self.image_label = QLabel("Select a folder, then run Search \u2192 Term.")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_scroll = QScrollArea()
        self.image_scroll.setWidgetResizable(True)
        self.image_scroll.setWidget(self.image_label)

        # Right: matching results.
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(4, 4, 4, 4)
        right_layout.addWidget(QLabel("Results"))
        self.results_list = QListWidget()
        self.results_list.currentItemChanged.connect(self._on_result_selected)
        right_layout.addWidget(self.results_list)

        splitter.addWidget(left)
        splitter.addWidget(self.image_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

        self.statusBar().showMessage("Ready")

    def _build_menu(self) -> None:
        menubar = self.menuBar()

        file_menu = menubar.addMenu("&File")
        select_action = QAction("Select Folder\u2026", self)
        select_action.triggered.connect(self.select_folder)
        file_menu.addAction(select_action)
        file_menu.addSeparator()
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        search_menu = menubar.addMenu("&Search")
        self.term_action = QAction("Term\u2026", self)
        self.term_action.triggered.connect(self.search_term)
        search_menu.addAction(self.term_action)
        self.stop_action = QAction("Stop", self)
        self.stop_action.setEnabled(False)
        self.stop_action.triggered.connect(self.stop_search)
        search_menu.addAction(self.stop_action)

        concurrency_menu = menubar.addMenu("&Concurrency")
        self.concurrency_group = QActionGroup(self)
        self.concurrency_group.setExclusive(True)
        options = (
            ("GPU Batching", CONCURRENCY_GPU_BATCH),
            ("Process Pool (Executor)", CONCURRENCY_EXECUTOR),
        )
        for label, value in options:
            action = QAction(label, self, checkable=True)
            action.setData(value)
            action.setChecked(value == self.concurrency)
            action.triggered.connect(self._on_concurrency_changed)
            self.concurrency_group.addAction(action)
            concurrency_menu.addAction(action)

        engine_menu = menubar.addMenu("OCR &Engine")
        self.engine_group = QActionGroup(self)
        self.engine_group.setExclusive(True)
        engine_options = (
            ("PaddleOCR", ENGINE_PADDLE),
            ("Tesseract", ENGINE_TESSERACT),
            ("RapidOCR", ENGINE_RAPIDOCR),
        )
        for label, value in engine_options:
            action = QAction(label, self, checkable=True)
            action.setData(value)
            action.setChecked(value == self.engine)
            action.triggered.connect(self._on_engine_changed)
            self.engine_group.addAction(action)
            engine_menu.addAction(action)

    # ------------------------------------------------------------- folder load
    def select_folder(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        start = self.folder or self.settings.value("lastFolder", "", type=str)
        folder = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if folder:
            self.load_folder(folder)

    def load_folder(self, folder: str) -> None:
        self.folder = folder
        self.settings.setValue("lastFolder", folder)

        self.pdf_paths = []
        for root, _, files in os.walk(folder):
            for name in files:
                if name.lower().endswith(".pdf"):
                    self.pdf_paths.append(os.path.join(root, name))
        self.pdf_paths.sort()

        # A new folder invalidates the OCR text cache and any prior results.
        self.text_cache.clear()
        self.results_list.clear()
        self.file_list.clear()
        for path in self.pdf_paths:
            item = QListWidgetItem(os.path.relpath(path, folder))
            item.setData(Qt.ItemDataRole.UserRole, path)
            self.file_list.addItem(item)

        self.image_label.setText("")
        self.image_label.setPixmap(QPixmap())
        self.statusBar().showMessage(
            f"{len(self.pdf_paths)} PDF(s) found in {folder}"
        )

    # ------------------------------------------------------------------ search
    def search_term(self) -> None:
        if not self.pdf_paths:
            QMessageBox.information(self, "No PDFs", "Select a folder with PDFs first.")
            return
        if self.thread is not None:
            return

        last_term = self.settings.value("lastTerm", "", type=str)
        term, ok = QInputDialog.getText(
            self, "Search Term", "Term to search for:", text=last_term
        )
        term = term.strip()
        if not ok or not term:
            return

        self.settings.setValue("lastTerm", term)
        self.results_list.clear()
        self.setWindowTitle(f"OCR PDF Folder Search \u2014 '{term}'")

        # Create the mirrored text-cache folder as the search begins.
        text_dir = text_dir_for(self.folder)
        try:
            os.makedirs(text_dir, exist_ok=True)
        except OSError as exc:
            QMessageBox.critical(
                self, "Cache error", f"Could not create '{text_dir}':\n{exc}"
            )
            return

        # Unparented so it isn't kept alive as a child of MainWindow; teardown
        # releases it explicitly via deleteLater.
        self.thread = QThread()
        self.worker = SearchWorker(
            list(self.pdf_paths),
            term,
            self.folder,
            text_dir,
            self.text_cache,
            self.concurrency,
            self.engine,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.match_found.connect(self._on_match_found)
        self.worker.progress.connect(self._on_progress)
        self.worker.failed.connect(self._on_failed)
        self.worker.fatal.connect(self._on_fatal)
        self.worker.finished.connect(self._on_finished)
        self.worker.status.connect(self._on_status)

        self.term_action.setEnabled(False)
        self.stop_action.setEnabled(True)
        self.statusBar().showMessage(f"Searching for '{term}'\u2026")
        self.thread.start()

    def stop_search(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.statusBar().showMessage("Stopping\u2026")

    def _on_concurrency_changed(self) -> None:
        action = self.concurrency_group.checkedAction()
        if action is None:
            return
        self.concurrency = action.data()
        self.settings.setValue("concurrency", self.concurrency)
        self.statusBar().showMessage(f"Concurrency: {action.text()}")

    def _on_engine_changed(self) -> None:
        action = self.engine_group.checkedAction()
        if action is None:
            return
        self.engine = action.data()
        self.settings.setValue("engine", self.engine)
        self.statusBar().showMessage(f"OCR engine: {action.text()}")

    # ---------------------------------------------------------- worker signals
    def _on_match_found(self, path: str) -> None:
        item = QListWidgetItem(os.path.relpath(path, self.folder or ""))
        item.setData(Qt.ItemDataRole.UserRole, path)
        self.results_list.addItem(item)

    def _on_progress(self, done: int, total: int) -> None:
        matches = self.results_list.count()
        self.statusBar().showMessage(
            f"Scanned {done}/{total} \u2014 {matches} match(es)"
        )

    def _on_failed(self, path: str, message: str) -> None:
        rel = os.path.relpath(path, self.folder or "")
        self.statusBar().showMessage(f"Failed on {rel}: {message}")

    def _on_status(self, message: str) -> None:
        self.statusBar().showMessage(message)

    def _on_fatal(self, message: str) -> None:
        QMessageBox.critical(self, "OCR error", message)
        self._teardown_thread()
        self.statusBar().showMessage("Search aborted")

    def _on_finished(self, matches: int) -> None:
        self._teardown_thread()
        self.statusBar().showMessage(f"Done \u2014 {matches} match(es)")

    def _teardown_thread(self) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait()
            self.thread.deleteLater()
        if self.worker is not None:
            self.worker.deleteLater()
        self.thread = None
        self.worker = None
        self.term_action.setEnabled(True)
        self.stop_action.setEnabled(False)

    # --------------------------------------------------------------- selection
    def _on_file_selected(
        self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None
    ) -> None:
        if item is None or self._syncing_selection:
            return
        self._display_path(item.data(Qt.ItemDataRole.UserRole))

    def _on_result_selected(
        self, item: QListWidgetItem | None, _previous: QListWidgetItem | None = None
    ) -> None:
        if item is None:
            return
        path = item.data(Qt.ItemDataRole.UserRole)
        self._select_in_file_list(path)
        self._display_path(path)

    def _display_path(self, path: str) -> None:
        try:
            image, _ = render_front_page(path)
            self._show_image(image)
        except Exception as exc:  # noqa: BLE001
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"Could not render:\n{exc}")

    def _show_image(self, image: QImage) -> None:
        self.image_label.setText("")
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def _select_in_file_list(self, path: str) -> None:
        for row in range(self.file_list.count()):
            item = self.file_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                # Suppress the file list's own selection handler so we don't
                # render the same page twice (the result handler renders it).
                self._syncing_selection = True
                try:
                    self.file_list.setCurrentItem(item)
                    self.file_list.scrollToItem(item)
                finally:
                    self._syncing_selection = False
                break

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self.worker is not None:
            self.worker.stop()
        self._teardown_thread()
        super().closeEvent(event)


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
