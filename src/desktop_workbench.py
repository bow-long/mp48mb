from __future__ import annotations

import ctypes
import base64
import csv
import io
import json
import math
import os
import queue
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import ctypes
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk


ROOT = Path(__file__).resolve().parents[1]
if getattr(sys, "frozen", False):
    DEFAULT_BACKEND = Path(sys._MEIPASS) / "build" / "mp48mb-portable.exe"
else:
    DEFAULT_BACKEND = ROOT / "build" / ("mp48mb.exe" if os.name == "nt" else "mp48mb")
MAX_INPUT_BYTES = 2 * 1024 * 1024 * 1024
VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm"}
IMAGE_SUFFIXES = {".ppm", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PROFILE_LABELS = {
    "source_decode_ms": "video decode",
    "color_convert_ms": "BGR-to-RGB conversion",
    "backend_input_prepare_ms": "backend input preparation",
    "backend_pipe_write_ms": "frame transfer to backend",
    "backend_response_pipe_ms": "backend response/pipe overhead",
    "codec_encode_ms": "MP48MB encode",
    "codec_decode_ms": "MP48MB decode",
    "quality_metrics_ms": "MSE/PSNR calculation",
    "sequence_frame_write_ms": "sequence frame write",
    "other_ms": "other per-frame work",
}


def write_ppm(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    with path.open("wb") as output:
        output.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        output.write(np.ascontiguousarray(rgb).tobytes())


def read_ppm(path: Path) -> np.ndarray:
    with path.open("rb") as source:
        if source.readline().strip() != b"P6":
            raise ValueError("Codec output is not a binary PPM frame")
        dimensions = source.readline()
        while dimensions.startswith(b"#"):
            dimensions = source.readline()
        width, height = (int(value) for value in dimensions.split())
        if source.readline().strip() != b"255":
            raise ValueError("Codec output PPM has an unsupported pixel range")
        pixels = source.read()
    expected = width * height * 3
    if len(pixels) != expected:
        raise ValueError("Codec output PPM is truncated")
    return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3).copy()


def run_backend(backend: Path, arguments: list[str], cancel: threading.Event) -> str:
    process = subprocess.Popen(
        [str(backend), *arguments],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    while process.poll() is None:
        if cancel.wait(0.05):
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise InterruptedError("Cancelled")
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        detail = (stderr or stdout).strip()
        raise RuntimeError(detail or f"Codec backend exited with code {process.returncode}")
    return stdout.strip()


def call_backend(backend: Path, arguments: list[str], cancel: threading.Event) -> tuple[float, str]:
    start = time.perf_counter()
    output = run_backend(backend, arguments, cancel)
    return (time.perf_counter() - start) * 1000.0, output


class BackendServer:
    _PARTITION_IDS = {"auto": 0, "single": 1, "four": 2}
    _MAX_ENCODED_BYTES = 256 * 1024 * 1024

    def __init__(self, backend: Path, cancel: threading.Event) -> None:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [str(backend), "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=creation_flags,
        )
        self.cancel = cancel
        if os.name == "nt":
            self._peek_named_pipe = ctypes.windll.kernel32.PeekNamedPipe
            self._peek_named_pipe.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_ulong,
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_ulong),
                ctypes.c_void_p,
            ]
            self._peek_named_pipe.restype = ctypes.c_int

    def _read_exact(self, size: int) -> bytes:
        assert self.process.stdout is not None
        stream = self.process.stdout
        result = bytearray()
        while len(result) < size:
            if self.cancel.is_set():
                self.terminate()
                raise InterruptedError("Cancelled")
            if self.process.poll() is not None:
                detail = b""
                if self.process.stderr is not None:
                    detail = self.process.stderr.read()
                raise RuntimeError(
                    detail.decode("utf-8", errors="replace").strip()
                    or f"Codec backend exited with code {self.process.returncode}"
                )
            if os.name == "nt":
                available = ctypes.c_ulong()
                import msvcrt

                ok = self._peek_named_pipe(
                    ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno())),
                    None,
                    0,
                    None,
                    ctypes.byref(available),
                    None,
                )
                if not ok:
                    time.sleep(0.001)
                    continue
                if available.value == 0:
                    time.sleep(0.001)
                    continue
            else:
                readable, _, _ = select.select([stream], [], [], 0.05)
                if not readable:
                    continue
            chunk = stream.read(size - len(result))
            if not chunk:
                raise RuntimeError("Codec backend closed its response stream unexpectedly")
            result.extend(chunk)
        return bytes(result)

    def _write_all(self, data: memoryview) -> None:
        assert self.process.stdin is not None
        offset = 0
        while offset < len(data):
            if self.cancel.is_set():
                self.terminate()
                raise InterruptedError("Cancelled")
            written = self.process.stdin.write(data[offset:])
            if written is None or written == 0:
                raise RuntimeError("Codec backend stopped accepting frame data")
            offset += written

    def encode_decode(
        self, rgb: np.ndarray, options: dict[str, Any]
    ) -> tuple[float, float, bytes, np.ndarray, dict[str, float]]:
        if self.cancel.is_set():
            raise InterruptedError("Cancelled")
        height, width, _ = rgb.shape
        if options["regions"] not in self._PARTITION_IDS:
            raise ValueError(f"Unsupported partition mode: {options['regions']}")
        if self.process.stdin is None:
            raise RuntimeError("Codec backend request stream is unavailable")
        header = struct.pack(
            "<7I",
            width,
            height,
            options["max_block"],
            options["min_block"],
            options["max_depth"],
            options["threshold"],
            self._PARTITION_IDS[options["regions"]],
        )
        profile: dict[str, float] = {}
        exchange_start = time.perf_counter()
        try:
            prepare_start = time.perf_counter()
            frame_bytes = memoryview(np.ascontiguousarray(rgb)).cast("B")
            profile["backend_input_prepare_ms"] = (time.perf_counter() - prepare_start) * 1000.0
            write_start = time.perf_counter()
            self._write_all(memoryview(header))
            self._write_all(frame_bytes)
            self.process.stdin.flush()
            profile["backend_pipe_write_ms"] = (time.perf_counter() - write_start) * 1000.0
            response_start = time.perf_counter()
            status = self._read_exact(1)[0]
            if status:
                message_size = struct.unpack("<I", self._read_exact(4))[0]
                message = self._read_exact(message_size).decode("utf-8", errors="replace")
                raise RuntimeError(message or "Codec backend rejected the frame")
            encode_ns, decode_ns, payload_size = struct.unpack("<QQI", self._read_exact(20))
            if payload_size > self._MAX_ENCODED_BYTES:
                raise RuntimeError(f"Codec backend returned an oversized frame ({payload_size} bytes)")
            payload = self._read_exact(payload_size)
            reconstructed_bytes = self._read_exact(width * height * 3)
            reconstructed = np.frombuffer(reconstructed_bytes, dtype=np.uint8).reshape(height, width, 3).copy()
            profile["backend_response_ms"] = (time.perf_counter() - response_start) * 1000.0
            profile["backend_roundtrip_ms"] = (time.perf_counter() - exchange_start) * 1000.0
            encode_ms = encode_ns / 1_000_000.0
            decode_ms = decode_ns / 1_000_000.0
            profile["backend_response_pipe_ms"] = max(
                0.0, profile["backend_response_ms"] - encode_ms - decode_ms
            )
            return encode_ms, decode_ms, payload, reconstructed, profile
        except (BrokenPipeError, OSError):
            if self.cancel.is_set():
                raise InterruptedError("Cancelled")
            raise RuntimeError("Lost communication with the codec backend") from None

    def terminate(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()

    def close(self) -> None:
        if self.process.poll() is None:
            if self.process.stdin is not None:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.terminate()
                self.process.wait()
        if self.process.stderr is not None:
            self.process.stderr.close()
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self) -> "BackendServer":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        if exc_type is not None or self.cancel.is_set():
            self.terminate()
        self.close()


def thumbnail_png(rgb: np.ndarray, max_side: int = 640) -> bytes:
    image = Image.fromarray(rgb, mode="RGB")
    image.thumbnail((max_side, max_side))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def psnr_from_mse(mse: float) -> float:
    return math.inf if mse == 0.0 else 10.0 * math.log10(255.0 * 255.0 / mse)


def frame_mse(source: np.ndarray, reconstructed: np.ndarray) -> float:
    squared_error = 0
    for row in range(source.shape[0]):
        difference = source[row].astype(np.int16) - reconstructed[row].astype(np.int16)
        squared_error += int(np.sum(difference.astype(np.int32) * difference.astype(np.int32), dtype=np.int64))
    return squared_error / (3.0 * source.shape[0] * source.shape[1])


class VideoWorkbench:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("MP48MB Video Workbench")
        self.root.geometry("1120x760")
        self.root.minsize(900, 620)

        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=2)
        self.cancel_event = threading.Event()
        self.worker: threading.Thread | None = None
        self.selected_path: Path | None = None
        self.backend_path = tk.StringVar(value=str(DEFAULT_BACKEND))
        self.output_path = tk.StringVar()
        self.region_mode = tk.StringVar(value="auto")
        self.max_block = tk.StringVar(value="256")
        self.min_block = tk.StringVar(value="8")
        self.max_depth = tk.StringVar(value="12")
        self.threshold = tk.StringVar(value="24")
        self.max_frames = tk.StringVar(value="1800")
        self.max_dimension = tk.StringVar(value="4096")
        self.detailed_profiling = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Choose or drop an image/video to begin.")
        self.metrics = tk.StringVar(value="No job has run.")
        self.original_photo: ImageTk.PhotoImage | None = None
        self.reconstructed_photo: ImageTk.PhotoImage | None = None
        self._drop_hook: NativeDropHook | None = None

        self._build_ui()
        self.root.after(60, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(200, self._enable_windows_drop)

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        header = ttk.Frame(self.root, padding=12)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(header, text="MP48MB Video Workbench", font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 10)
        )
        ttk.Button(header, text="Open image / video…", command=self._choose_file).grid(row=1, column=0, sticky="w")
        self.file_label = ttk.Label(header, text="Drop a supported file here or use Open.")
        self.file_label.grid(row=1, column=1, sticky="ew", padx=10)
        ttk.Button(header, text="Save output as…", command=self._choose_output).grid(row=1, column=2, sticky="e")

        controls = ttk.LabelFrame(self.root, text="Encoding controls and resource limits", padding=10)
        controls.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        for column in range(12):
            controls.columnconfigure(column, weight=1 if column in (1, 3, 5, 7, 9, 11) else 0)

        entries = [
            ("Regions", self.region_mode, 0, ("auto", "single", "four")),
            ("Max block", self.max_block, 2, None),
            ("Min block", self.min_block, 4, None),
            ("Max depth", self.max_depth, 6, None),
            ("MSE threshold", self.threshold, 8, None),
        ]
        for label, variable, column, choices in entries:
            ttk.Label(controls, text=label).grid(row=0, column=column, sticky="w")
            if choices:
                ttk.Combobox(controls, textvariable=variable, values=choices, state="readonly", width=9).grid(
                    row=0, column=column + 1, sticky="ew", padx=(4, 10)
                )
            else:
                ttk.Entry(controls, textvariable=variable, width=9).grid(
                    row=0, column=column + 1, sticky="ew", padx=(4, 10)
                )
        ttk.Label(controls, text="Max frames").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.max_frames, width=9).grid(row=1, column=1, sticky="ew", padx=(4, 10), pady=(8, 0))
        ttk.Label(controls, text="Max dimension").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.max_dimension, width=9).grid(row=1, column=3, sticky="ew", padx=(4, 10), pady=(8, 0))
        ttk.Label(controls, text="Backend").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(controls, textvariable=self.backend_path).grid(row=1, column=5, columnspan=5, sticky="ew", padx=(4, 10), pady=(8, 0))
        ttk.Button(controls, text="Browse…", command=self._choose_backend).grid(row=1, column=10, sticky="ew", pady=(8, 0))
        ttk.Checkbutton(
            controls, text="Detailed profiling (write per-frame CSV)", variable=self.detailed_profiling
        ).grid(row=2, column=0, columnspan=5, sticky="w", pady=(8, 0))

        body = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        body.grid(row=2, column=0, sticky="nsew", padx=12, pady=8)
        self.original_panel, self.reconstructed_panel = ttk.LabelFrame(body, text="Original", padding=8), ttk.LabelFrame(
            body, text="Reconstructed", padding=8
        )
        self.original_panel.columnconfigure(0, weight=1)
        self.original_panel.rowconfigure(0, weight=1)
        self.reconstructed_panel.columnconfigure(0, weight=1)
        self.reconstructed_panel.rowconfigure(0, weight=1)
        body.add(self.original_panel, weight=1)
        body.add(self.reconstructed_panel, weight=1)
        self.original_view = ttk.Label(self.original_panel, text="Original preview")
        self.original_view.grid(row=0, column=0, sticky="nsew")
        self.reconstructed_view = ttk.Label(self.reconstructed_panel, text="Reconstructed preview")
        self.reconstructed_view.grid(row=0, column=0, sticky="nsew")

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(footer, mode="determinate", maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        button_row = ttk.Frame(footer)
        button_row.grid(row=1, column=0, sticky="ew")
        self.start_button = ttk.Button(button_row, text="Encode", command=self._start)
        self.start_button.pack(side=tk.LEFT)
        self.cancel_button = ttk.Button(button_row, text="Cancel", command=self._cancel, state=tk.DISABLED)
        self.cancel_button.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(button_row, textvariable=self.status).pack(side=tk.LEFT, padx=12)
        ttk.Label(footer, textvariable=self.metrics, justify=tk.LEFT, wraplength=1040).grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )

    def _enable_windows_drop(self) -> None:
        if os.name == "nt":
            try:
                self._drop_hook = NativeDropHook(self.root, self._on_drop)
            except (AttributeError, OSError) as error:
                self.status.set(f"Native drag-and-drop unavailable; use Open. ({error})")

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Open image or video",
            filetypes=[
                ("Supported media", "*.mp4 *.mov *.m4v *.mkv *.avi *.webm *.ppm *.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._set_input(Path(path))

    def _choose_output(self) -> None:
        if self.selected_path is None:
            messagebox.showinfo("Select input", "Choose an image or video before selecting an output.")
            return
        video = self.selected_path.suffix.lower() in VIDEO_SUFFIXES
        suffix = ".mp48seq" if video else ".mpf"
        path = filedialog.asksaveasfilename(
            title="Save MP48MB output",
            defaultextension=suffix,
            initialfile=self.selected_path.stem + suffix,
            filetypes=[("MP48MB custom sequence" if video else "MP48MB frame", f"*{suffix}"), ("All files", "*.*")],
        )
        if path:
            self.output_path.set(path)

    def _choose_backend(self) -> None:
        path = filedialog.askopenfilename(title="Select MP48MB CLI backend", filetypes=[("Executable", "*.exe"), ("All files", "*.*")])
        if path:
            self.backend_path.set(path)

    def _on_drop(self, paths: list[str]) -> None:
        if paths:
            self._set_input(Path(paths[0]))

    def _set_input(self, path: Path) -> None:
        if path.suffix.lower() not in VIDEO_SUFFIXES | IMAGE_SUFFIXES:
            messagebox.showerror("Unsupported input", "Choose an MP4/video or a supported still image.")
            return
        self.selected_path = path
        self.file_label.configure(text=str(path))
        if path.suffix.lower() in VIDEO_SUFFIXES:
            self.original_photo = None
            self.reconstructed_photo = None
            self.original_view.configure(image="", text="Video preview disabled")
            self.reconstructed_view.configure(image="", text="Video preview disabled")
        else:
            self.original_photo = None
            self.reconstructed_photo = None
            self.original_view.configure(image="", text="Original preview")
            self.reconstructed_view.configure(image="", text="Reconstructed preview")
        suffix = ".mp48seq" if path.suffix.lower() in VIDEO_SUFFIXES else ".mpf"
        self.output_path.set(str(path.with_name(path.stem + suffix)))
        self.status.set("Ready to encode.")
        self.progress["value"] = 0

    def _start(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        if self.selected_path is None:
            self._choose_file()
            if self.selected_path is None:
                return
        backend = Path(self.backend_path.get())
        if not backend.is_file():
            messagebox.showerror("Backend not found", f"Build the project first, then select its executable:\n{backend}")
            return
        try:
            options = self._options()
        except ValueError as error:
            messagebox.showerror("Invalid settings", str(error))
            return
        output = Path(self.output_path.get())
        if not output.parent.exists():
            messagebox.showerror("Invalid output", "The output directory does not exist.")
            return

        self.cancel_event.clear()
        self.start_button.configure(state=tk.DISABLED)
        self.cancel_button.configure(state=tk.NORMAL)
        self.progress["value"] = 0
        self.status.set("Starting…")
        self.worker = threading.Thread(
            target=self._worker_main,
            args=(self.selected_path, output, backend, options),
            name="mp48mb-workbench",
            daemon=True,
        )
        self.worker.start()

    def _options(self) -> dict[str, Any]:
        def positive(name: str, text: str, upper: int) -> int:
            try:
                value = int(text)
            except ValueError as error:
                raise ValueError(f"{name} must be an integer.") from error
            if not 1 <= value <= upper:
                raise ValueError(f"{name} must be between 1 and {upper}.")
            return value

        max_block = positive("Max block", self.max_block.get(), 4096)
        min_block = positive("Min block", self.min_block.get(), 4096)
        max_depth = positive("Max depth", self.max_depth.get(), 12)
        try:
            threshold = int(self.threshold.get())
        except ValueError as error:
            raise ValueError("MSE threshold must be an integer.") from error
        if not 0 <= threshold <= 65025:
            raise ValueError("MSE threshold must be between 0 and 65025.")
        max_frames = positive("Max frames", self.max_frames.get(), 100000)
        max_dimension = positive("Max dimension", self.max_dimension.get(), 4096)
        for name, value in (("Max block", max_block), ("Min block", min_block)):
            if value & (value - 1):
                raise ValueError(f"{name} must be a power of two.")
        if min_block > max_block:
            raise ValueError("Min block cannot exceed max block.")
        return {
            "max_block": max_block,
            "min_block": min_block,
            "max_depth": max_depth,
            "threshold": threshold,
            "max_frames": max_frames,
            "max_dimension": max_dimension,
            "regions": self.region_mode.get(),
            "detailed_profiling": self.detailed_profiling.get(),
        }

    def _post(self, event: dict[str, Any]) -> None:
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            self.events.put_nowait(event)

    def _worker_main(self, source: Path, output: Path, backend: Path, options: dict[str, Any]) -> None:
        self._post({"type": "start", "source": str(source), "output": str(output)})
        try:
            if source.stat().st_size > MAX_INPUT_BYTES:
                raise ValueError(f"Input exceeds the {MAX_INPUT_BYTES // (1024 * 1024)} MiB file-size limit.")
            if source.suffix.lower() in VIDEO_SUFFIXES:
                summary = self._process_video(source, output, backend, options)
            else:
                summary = self._process_image(source, output, backend, options)
            if not self.cancel_event.is_set():
                self._post({"type": "complete", "summary": summary})
            else:
                self._post({"type": "cancelled"})
        except InterruptedError:
            self._post({"type": "cancelled"})
        except Exception as error:
            self._post({"type": "error", "message": str(error)})

    def _codec_args(self, options: dict[str, Any]) -> list[str]:
        return [
            "--max-block", str(options["max_block"]),
            "--min-block", str(options["min_block"]),
            "--max-depth", str(options["max_depth"]),
            "--threshold", str(options["threshold"]),
            "--regions", options["regions"],
        ]

    def _process_image(self, source: Path, output: Path, backend: Path, options: dict[str, Any]) -> dict[str, Any]:
        self._post({"type": "stage", "stage": "decoding still image"})
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Unable to decode the selected image.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, _ = rgb.shape
        self._validate_dimensions(width, height, options)
        self._post({
            "type": "progress", "frame": 1, "total": 1, "width": width, "height": height,
            "fps": 0.0, "source_decode_ms": 0.0, "codec_encode_ms": 0.0,
            "codec_decode_ms": 0.0, "bytes": 0, "mse": 0.0,
            "region_mode": self._region_label(width, height, options["regions"]), "workers": 1,
            "preview_original": thumbnail_png(rgb),
        })
        with tempfile.TemporaryDirectory(prefix="mp48mb-") as temp_name:
            temp = Path(temp_name)
            input_ppm = temp / "input.ppm"
            encoded_file = temp / "frame.mpf"
            reconstructed_ppm = temp / "reconstructed.ppm"
            write_ppm(input_ppm, rgb)
            self._post({"type": "stage", "stage": "MP48MB encode"})
            encode_ms, _ = call_backend(backend, ["encode", str(input_ppm), str(encoded_file), *self._codec_args(options)], self.cancel_event)
            payload = encoded_file.read_bytes()
            self._post({"type": "stage", "stage": "MP48MB decode"})
            decode_ms, _ = call_backend(backend, ["decode", str(encoded_file), str(reconstructed_ppm)], self.cancel_event)
            reconstructed = read_ppm(reconstructed_ppm)
            mse = frame_mse(rgb, reconstructed)
            self._atomic_write(output, payload)
            self._post({
                "type": "progress", "frame": 1, "total": 1, "width": width, "height": height,
                "fps": 0.0, "source_decode_ms": 0.0, "codec_encode_ms": encode_ms,
                "codec_decode_ms": decode_ms, "bytes": len(payload), "mse": mse,
                "region_mode": self._region_label(width, height, options["regions"]), "workers": 1,
                "preview_original": thumbnail_png(rgb), "preview_reconstructed": thumbnail_png(reconstructed),
            })
            return {
                "kind": "frame", "path": str(output), "width": width, "height": height,
                "encoded_bytes": len(payload), "mse": mse, "psnr": psnr_from_mse(mse),
                "encode_ms": encode_ms, "decode_ms": decode_ms,
                "processing_fps": 1000.0 / max(encode_ms + decode_ms, 1e-9),
                "region_mode": self._region_label(width, height, options["regions"]),
                "worker_count": 1, "audio": "not applicable",
            }

    def _process_video(self, source: Path, output: Path, backend: Path, options: dict[str, Any]) -> dict[str, Any]:
        job_started_at = time.perf_counter()
        capture = cv2.VideoCapture(str(source), cv2.CAP_FFMPEG)
        if not capture.isOpened():
            capture.release()
            raise ValueError("OpenCV/FFmpeg could not open this video. Check the codec and file integrity.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            self._validate_dimensions(width, height, options)
            if reported_total > options["max_frames"]:
                raise ValueError(f"Video reports {reported_total} frames, exceeding the configured limit of {options['max_frames']}.")
            audio_status = self._audio_status(source)
            metadata = {
                "format": "MP48MB custom frame sequence",
                "version": 1,
                "width": width,
                "height": height,
                "pixel_format": "RGB24",
                "input_fps": fps if math.isfinite(fps) and fps > 0 else None,
                "frame_timestamps": "microseconds from OpenCV CAP_PROP_POS_MSEC; frame order preserved",
                "audio_present": audio_status,
                "audio_policy": "omitted; this is not an MP4 container",
                "region_mode": options["regions"],
                "codec_version": 3,
            }

            frames_processed = 0
            total_payload_bytes = 0
            source_decode_ms_total = 0.0
            encode_ms_total = 0.0
            codec_decode_ms_total = 0.0
            mse_sum = 0.0
            started_at = time.perf_counter()
            profile_totals_ms = {
                name: 0.0
                for name in (
                    "source_decode_ms",
                    "color_convert_ms",
                    "backend_input_prepare_ms",
                    "backend_pipe_write_ms",
                    "backend_response_pipe_ms",
                    "codec_encode_ms",
                    "codec_decode_ms",
                    "quality_metrics_ms",
                    "sequence_frame_write_ms",
                    "other_ms",
                )
            }
            output.parent.mkdir(parents=False, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="mp48mb-") as temp_name:
                temp = Path(temp_name)
                body_path = temp / "frames.bin"
                with body_path.open("wb") as frame_body, BackendServer(backend, self.cancel_event) as codec_backend:
                    profile_file = (temp / "profile.csv").open("w", newline="", encoding="utf-8") \
                        if options.get("detailed_profiling") else None
                    profile_writer = csv.writer(profile_file) if profile_file is not None else None
                    if profile_writer is not None:
                        profile_writer.writerow(["frame", *profile_totals_ms.keys(), "frame_total_ms"])
                    try:
                        while True:
                            if self.cancel_event.is_set():
                                raise InterruptedError("Cancelled")
                            if frames_processed >= options["max_frames"]:
                                if reported_total <= 0 or frames_processed < reported_total:
                                    raise ValueError(f"Stopped at the configured limit of {options['max_frames']} frames.")
                                break

                            frame_started = time.perf_counter()
                            stages = {name: 0.0 for name in profile_totals_ms}
                            self._post({"type": "stage", "stage": f"decoding source frame {frames_processed + 1}"})
                            decode_start = time.perf_counter()
                            ok, bgr = capture.read()
                            source_decode_ms = (time.perf_counter() - decode_start) * 1000.0
                            if not ok:
                                break
                            frame_decode_ms = source_decode_ms
                            stages["source_decode_ms"] = source_decode_ms
                            if bgr.shape[1] != width or bgr.shape[0] != height:
                                raise ValueError("Video changed frame dimensions mid-stream; this input is not supported.")
                            conversion_start = time.perf_counter()
                            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                            stages["color_convert_ms"] = (time.perf_counter() - conversion_start) * 1000.0
                            timestamp_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                            if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
                                timestamp_ms = frames_processed * 1000.0 / fps if fps > 0 else 0.0
                            timestamp_us = max(0, int(round(timestamp_ms * 1000.0)))
                            self._post({"type": "stage", "stage": f"MP48MB processing frame {frames_processed + 1}"})
                            encode_ms, codec_decode_ms, payload, reconstructed, backend_profile = codec_backend.encode_decode(rgb, options)
                            stages["backend_input_prepare_ms"] = backend_profile["backend_input_prepare_ms"]
                            stages["backend_pipe_write_ms"] = backend_profile["backend_pipe_write_ms"]
                            stages["backend_response_pipe_ms"] = backend_profile["backend_response_pipe_ms"]
                            stages["codec_encode_ms"] = encode_ms
                            stages["codec_decode_ms"] = codec_decode_ms
                            quality_start = time.perf_counter()
                            mse = frame_mse(rgb, reconstructed)
                            stages["quality_metrics_ms"] = (time.perf_counter() - quality_start) * 1000.0
                            output_write_start = time.perf_counter()
                            frame_body.write(struct.pack("<qI", timestamp_us, len(payload)))
                            frame_body.write(payload)
                            stages["sequence_frame_write_ms"] = (time.perf_counter() - output_write_start) * 1000.0

                            frame_total_ms = (time.perf_counter() - frame_started) * 1000.0
                            stages["other_ms"] = max(0.0, frame_total_ms - sum(stages.values()))
                            for name, duration in stages.items():
                                profile_totals_ms[name] += duration

                            frames_processed += 1
                            total_payload_bytes += len(payload)
                            source_decode_ms_total += frame_decode_ms
                            encode_ms_total += encode_ms
                            codec_decode_ms_total += codec_decode_ms
                            mse_sum += mse
                            if profile_writer is not None:
                                profile_writer.writerow([
                                    frames_processed,
                                    *(f"{stages[name]:.6f}" for name in profile_totals_ms),
                                    f"{frame_total_ms:.6f}",
                                ])
                            self._post({
                                "type": "progress",
                                "frame": frames_processed,
                                "total": reported_total if reported_total > 0 else None,
                                "width": width,
                                "height": height,
                                "fps": fps,
                                "source_decode_ms": frame_decode_ms,
                                "codec_encode_ms": encode_ms,
                                "codec_decode_ms": codec_decode_ms,
                                "profile_top_stage": max(profile_totals_ms, key=profile_totals_ms.get),
                                "profile_top_stage_ms": max(profile_totals_ms.values()),
                                "bytes": total_payload_bytes,
                                "mse": mse,
                                "processing_fps": frames_processed / max(time.perf_counter() - started_at, 1e-9),
                                "region_mode": self._region_label(width, height, options["regions"]),
                                "workers": 1,
                            })
                    finally:
                        if profile_file is not None:
                            profile_file.close()

                if frames_processed == 0:
                    raise ValueError("The video contains no decodable frames.")
                metadata["frame_count"] = frames_processed
                metadata["audio_present"] = audio_status
                json_metadata = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
                temporary_output = temp / "result.mp48seq"
                assembly_start = time.perf_counter()
                with temporary_output.open("wb") as result:
                    result.write(b"MP48SEQ1")
                    result.write(struct.pack("<I", len(json_metadata)))
                    result.write(json_metadata)
                    with body_path.open("rb") as frame_body:
                        shutil.copyfileobj(frame_body, result, length=1024 * 1024)
                output_assembly_ms = (time.perf_counter() - assembly_start) * 1000.0
                if self.cancel_event.is_set():
                    raise InterruptedError("Cancelled")
                output_copy_start = time.perf_counter()
                self._atomic_copy(output, temporary_output)
                output_copy_ms = (time.perf_counter() - output_copy_start) * 1000.0
                profile_path: str | None = None
                if profile_file is not None:
                    profile_target = output.with_name(output.stem + ".profile.csv")
                    self._atomic_copy(profile_target, temp / "profile.csv")
                    profile_path = str(profile_target)

            return {
                "kind": "sequence",
                "path": str(output),
                "width": width,
                "height": height,
                "frame_count": frames_processed,
                "fps": fps,
                "encoded_bytes": output.stat().st_size,
                "payload_bytes": total_payload_bytes,
                "mse": mse_sum / frames_processed,
                "psnr": psnr_from_mse(mse_sum / frames_processed),
                "source_decode_ms": source_decode_ms_total,
                "encode_ms": encode_ms_total,
                "codec_decode_ms": codec_decode_ms_total,
                "processing_fps": frames_processed / max(time.perf_counter() - started_at, 1e-9),
                "region_mode": self._region_label(width, height, options["regions"]),
                "worker_count": 1,
                "audio": f"{audio_status}; omitted from output",
                "profile": profile_totals_ms,
                "profile_setup_ms": max(0.0, (started_at - job_started_at) * 1000.0),
                "profile_output_assembly_ms": output_assembly_ms,
                "profile_output_copy_ms": output_copy_ms,
                "profile_path": profile_path,
                "codec_bottleneck": (
                    "encode" if encode_ms_total > codec_decode_ms_total * 1.05
                    else "decode" if codec_decode_ms_total > encode_ms_total * 1.05
                    else "similar"
                ),
            }
        finally:
            capture.release()

    @staticmethod
    def _validate_dimensions(width: int, height: int, options: dict[str, Any]) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("Input reported invalid frame dimensions.")
        if width > options["max_dimension"] or height > options["max_dimension"]:
            raise ValueError(
                f"Frame dimensions {width}x{height} exceed the configured per-axis limit "
                f"of {options['max_dimension']} pixels."
            )
        if width * height > 16_777_216:
            raise ValueError("Frame exceeds the 16.8-megapixel allocation limit.")

    @staticmethod
    def _audio_status(source: Path) -> bool | None:
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            return None
        result = subprocess.run(
            [
                ffprobe, "-v", "error", "-select_streams", "a",
                "-show_entries", "stream=index", "-of", "csv=p=0", str(source),
            ],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        if result.returncode != 0:
            return None
        return bool(result.stdout.strip())

    @staticmethod
    def _region_label(width: int, height: int, selected: str) -> str:
        if selected == "four":
            return "four-region (sequential)"
        if selected == "single":
            return "single-region"
        return "four-region (sequential)" if width > 640 or height > 480 else "single-region"

    @staticmethod
    def _atomic_write(path: Path, content: bytes) -> None:
        temporary = path.with_name(path.name + ".partial")
        with temporary.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)

    @staticmethod
    def _atomic_copy(path: Path, source: Path) -> None:
            temporary = path.with_name(path.name + ".partial")
            shutil.copyfile(source, temporary)
            with temporary.open("rb+") as copied:
                os.fsync(copied.fileno())
            temporary.replace(path)

    def _cancel(self) -> None:
        self.cancel_event.set()
        self.status.set("Cancelling…")
        self.cancel_button.configure(state=tk.DISABLED)

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                kind = event["type"]
                if kind == "start":
                    self.status.set("Processing…")
                    self.metrics.set(f"Input: {event['source']}\nOutput: {event['output']}")
                elif kind == "progress":
                    total = event["total"]
                    frame = event["frame"]
                    if total:
                        self.progress["maximum"] = total
                        self.progress["value"] = frame
                    else:
                        self.progress["maximum"] = max(self.progress["maximum"], frame)
                        self.progress["value"] = frame
                    self.status.set(f"Frame {frame}" + (f" / {total}" if total else " / total unknown"))
                    self.metrics.set(
                        f"{event['width']}x{event['height']} at {event['fps']:.3f} fps | "
                        f"source decode {event['source_decode_ms']:.2f} ms | "
                        f"codec encode {event['codec_encode_ms']:.2f} ms | "
                        f"codec decode {event['codec_decode_ms']:.2f} ms | "
                        f"pipeline {event.get('processing_fps', 0.0):.2f} frames/s | "
                        f"cumulative MPF payload {event['bytes']:,} bytes | "
                        f"frame MSE {event['mse']:.3f}, PSNR {psnr_from_mse(event['mse']):.2f} dB | "
                        f"{event.get('region_mode', 'single-region')} | workers {event.get('workers', 1)}"
                    )
                    if event.get("profile_top_stage") is not None:
                        stage_name = PROFILE_LABELS[event["profile_top_stage"]]
                        self.metrics.set(
                            self.metrics.get()
                            + f"\nProfiling so far: {stage_name} is highest "
                            f"({event['profile_top_stage_ms']:.1f} ms accumulated)."
                        )
                    if "preview_original" in event or "preview_reconstructed" in event:
                        self._show_preview(self.original_view, event.get("preview_original"), "original")
                        self._show_preview(self.reconstructed_view, event.get("preview_reconstructed"), "reconstructed")
                elif kind == "stage":
                    self.status.set(event["stage"])
                elif kind == "complete":
                    data = event["summary"]
                    self.status.set("Completed.")
                    self.progress["value"] = self.progress["maximum"]
                    self.metrics.set(
                        f"Saved {data['path']} | {data.get('frame_count', 1)} frame(s) at "
                        f"{data['width']}x{data['height']} | {data['region_mode']} | "
                        f"{data['encoded_bytes']:,} output bytes | MSE {data['mse']:.3f}, "
                        f"PSNR {data['psnr']:.2f} dB | pipeline {data['processing_fps']:.2f} frames/s | "
                        f"source decode {data.get('source_decode_ms', 0):.2f} ms total | "
                        f"codec encode {data['encode_ms']:.2f} ms total | "
                        f"codec decode {data.get('decode_ms', data.get('codec_decode_ms', 0)):.2f} ms total | "
                        f"workers 1 | audio {data['audio']}"
                    )
                    if data["kind"] == "sequence":
                        self.metrics.set(
                            self.metrics.get()
                            + f"\nCodec bottleneck: {data['codec_bottleneck']} "
                            f"(encode {data['encode_ms']:.1f} ms vs decode "
                            f"{data['codec_decode_ms']:.1f} ms total). Video previews are disabled."
                        )
                        self.metrics.set(self.metrics.get() + "\nOutput is a custom MP48SEQ frame sequence, not MP4; audio is omitted.")
                        profile = data.get("profile")
                        if profile:
                            profile_total_ms = sum(profile.values())
                            ranked_stages = sorted(profile, key=profile.get, reverse=True)[:3]
                            self.metrics.set(
                                self.metrics.get()
                                + "\n\nProfiling totals (per-frame pipeline stages):"
                                + "".join(
                                    f"\n- {PROFILE_LABELS[name]}: {profile[name]:.1f} ms "
                                    f"({100.0 * profile[name] / max(profile_total_ms, 1e-9):.1f}%)"
                                    for name in ranked_stages
                                )
                                + f"\nSetup: {data['profile_setup_ms']:.1f} ms | "
                                + f"final sequence assembly: {data['profile_output_assembly_ms']:.1f} ms | "
                                + f"output copy: {data['profile_output_copy_ms']:.1f} ms"
                            )
                            if data.get("profile_path"):
                                self.metrics.set(self.metrics.get() + f"\nPer-frame breakdown saved to {data['profile_path']}")
                    self._set_idle()
                elif kind == "error":
                    self.status.set("Failed.")
                    self._set_idle()
                    messagebox.showerror("MP48MB job failed", event["message"])
                elif kind == "cancelled":
                    self.status.set("Cancelled.")
                    self._set_idle()
        except queue.Empty:
            pass
        self.root.after(60, self._drain_events)

    def _show_preview(self, widget: ttk.Label, data: bytes | None, which: str) -> None:
        if not data:
            return
        photo = ImageTk.PhotoImage(data=base64.b64encode(data).decode("ascii"), format="png")
        widget.configure(image=photo, text="")
        if which == "original":
            self.original_photo = photo
        else:
            self.reconstructed_photo = photo

    def _set_idle(self) -> None:
        self.start_button.configure(state=tk.NORMAL)
        self.cancel_button.configure(state=tk.DISABLED)
        self.worker = None

    def _close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.cancel_event.set()
            self.root.after(100, self._close_when_done)
            return
        self.root.destroy()

    def _close_when_done(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.root.after(100, self._close_when_done)
        else:
            self.root.destroy()


class NativeDropHook:
    WM_DROPFILES = 0x0233
    GWL_WNDPROC = -4
    DROP_QUERY_FILE_COUNT = 0xFFFFFFFF

    def __init__(self, root: tk.Tk, on_drop: Any) -> None:
        self.root = root
        self.on_drop = on_drop
        self.shell32 = ctypes.windll.shell32
        self.user32 = ctypes.windll.user32
        self.hwnd = root.winfo_id()
        self.wndproc_type = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t
        )
        self.user32.CallWindowProcW.restype = ctypes.c_ssize_t
        self.user32.CallWindowProcW.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_size_t, ctypes.c_ssize_t
        ]
        self.user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        self.user32.GetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        self.user32.SetWindowLongPtrW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_ssize_t]
        self.shell32.DragAcceptFiles.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        self.shell32.DragQueryFileW.restype = ctypes.c_uint
        self.shell32.DragQueryFileW.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_wchar), ctypes.c_uint
        ]
        self.shell32.DragFinish.argtypes = [ctypes.c_void_p]
        self.old_proc = self.user32.GetWindowLongPtrW(self.hwnd, self.GWL_WNDPROC)
        self.callback = self.wndproc_type(self._dispatch)
        self.user32.SetWindowLongPtrW(self.hwnd, self.GWL_WNDPROC, ctypes.cast(self.callback, ctypes.c_void_p).value)
        self.shell32.DragAcceptFiles(self.hwnd, True)

    def _dispatch(self, hwnd: int, message: int, wparam: int, lparam: int) -> int:
        if message == self.WM_DROPFILES:
            count = self.shell32.DragQueryFileW(wparam, self.DROP_QUERY_FILE_COUNT, None, 0)
            paths: list[str] = []
            for index in range(count):
                length = self.shell32.DragQueryFileW(wparam, index, None, 0)
                buffer = ctypes.create_unicode_buffer(length + 1)
                self.shell32.DragQueryFileW(wparam, index, buffer, length + 1)
                paths.append(buffer.value)
            self.shell32.DragFinish(wparam)
            self.root.after(0, lambda: self.on_drop(paths))
            return 0
        return int(self.user32.CallWindowProcW(self.old_proc, hwnd, message, wparam, lparam))


def main() -> None:
    root = tk.Tk()
    VideoWorkbench(root)
    root.mainloop()


if __name__ == "__main__":
    main()
