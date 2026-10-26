# MP48MB Video Workbench

MP48MB is a C++20 CPU-only quadtree frame codec prototype with a Tkinter desktop workbench. The workbench calls the C++ CLI for all MP48MB encoding and decoding; its Python code handles file selection, OpenCV/FFmpeg video input, previews, job control, and telemetry.

## Build and run

On Windows with MinGW UCRT64:

```powershell
C:\msys64\ucrt64\bin\mingw32-make.exe all test
python -m pip install -r requirements.txt
python src\desktop_workbench.py
```

`make gui` builds the C++ executable and starts the desktop workbench when `make` and Python are on `PATH`.

The GUI requires Python 3.10+, Tkinter, OpenCV with FFmpeg support, NumPy, and Pillow. Tkinter is usually included with the Windows Python installer. OpenCV provides the FFmpeg-backed video decoder dependency; no separate FFmpeg executable is required for decoding. If `ffprobe` is on `PATH`, the workbench checks whether an audio stream exists; otherwise audio presence is reported as unknown. The workbench does not preserve audio.

## Build a single-file Windows executable

Install the Python dependencies and the packaging tool, then run the PowerShell packaging script from the project directory:

```powershell
python -m pip install -r requirements-build.txt
.\package-exe.ps1
```

The script builds a C++ backend with the MinGW C++ runtime linked statically, bundles that backend and the GUI dependencies, and produces `dist\MP48MB.exe`. Copy and run that one file; it does not require a separate Python installation or `build\mp48mb.exe`. The executable unpacks its bundled backend into a temporary runtime directory when launched. MinGW UCRT64 and Python with Tkinter are required only to build it. This is a Windows build workflow; create the executable on Windows.

## Desktop workbench

The Windows UI supports native file drag-and-drop and a file picker for MP4 and other OpenCV-supported videos, plus common still-image formats. It shows original and reconstructed previews, encoding controls, progress and current stage, source frame rate, current frame index, source decode time, codec encode/decode time, pipeline throughput, actual cumulative MPF payload bytes, and per-frame MSE/PSNR. Video jobs start one persistent C++ backend process and stream RGB frames to it over binary pipes; no per-frame CLI launches or PPM temporary files are used. Codec timings are measured inside the C++ encoder/decoder and do not include pipe transfer or Python preview/output work. Jobs run on one background worker; cancellation is available between frames and terminates the backend if a frame is in progress.

Video previews are disabled during video processing to avoid thumbnail generation and UI redraw work. The preview panels are cleared when a video is selected; still-image previews continue to work.

Enable **Detailed profiling (write per-frame CSV)** to save `<output-name>.profile.csv` beside the encoded sequence. The completion summary ranks the largest cumulative per-frame stages and compares C++ encode and decode totals to identify which codec stage took longer. The CSV has per-frame timings for video decode, color conversion, backend input preparation and pipe transfer, C++ encode/decode, quality metrics, sequence writes, and other measured frame work. Backend response/pipe overhead includes receiving the encoded payload and reconstructed RGB frame. Setup and final output assembly/copy are reported separately. This identifies whether the current bottleneck is decoding, codec work, data transfer, or packaging overhead; compare runs with the same input, settings, and machine. Profiling adds a small amount of timing and CSV-write overhead and is intended for diagnosis rather than final throughput benchmarks.

Resource limits are explicit: a 2 GiB input-file limit, a user-configurable maximum frame count (default 1800), a configurable per-axis dimension limit (default 4096), and a 16.8-megapixel frame allocation cap. Video is decoded one frame at a time. The source color conversion is OpenCV/FFmpeg BGR to 8-bit RGB24; HDR, high-bit-depth and color-profile preservation are not implemented.

Still-image output is a custom `.mpf` frame. Video output is a custom `.mp48seq` sequence, not MP4. The sequence contains JSON metadata (dimensions, RGB24, frame rate, timestamps, frame count, audio status) followed by ordered timestamp/payload records; every payload is an independently decodable MPF frame. Timestamps come from OpenCV's `CAP_PROP_POS_MSEC` and are stored in microseconds. Audio is omitted. The output must not be described as a standards-compliant MP4; conventional MP4 export would need a standard video encoder and muxer.

## CLI

Encode a still PPM frame:

```powershell
.\build\mp48mb.exe encode input.ppm output.mpf --max-block 256 --min-block 8 --max-depth 12 --threshold 24 --regions auto
```

`--regions auto` uses single-region encoding when width <= 640 and height <= 480, and four-region encoding when either dimension exceeds that threshold. `--regions single` and `--regions four` force a mode for comparison. Four-region mode splits at floor(width/2), floor(height/2); stored region metadata and clipped edge blocks preserve odd dimensions.

Decode a frame:

```powershell
.\build\mp48mb.exe decode output.mpf reconstructed.ppm
```

The decoder accepts MPF bitstream versions 1 and 2 and the current version 3. Version 3 validates region metadata, leaf flags, overlap, coverage, truncation, and trailing bytes. Temporal prediction requires passing the same previous reference frame to both the encoder and decoder; the desktop video workflow currently encodes independent frames.

## Experimental fixed-block RGB565 codec

The fixed-block mode is a separate experiment and does not change the MPF quadtree format or its decoder. It stores one RGB565 average color for each 8x8 block and reconstructs each block as a solid color. RGB565 is packed little-endian; partial blocks at odd image edges average only pixels that exist. Its `MP48B565` header records version, block size, pixel format, width, and height. The decoder rejects unsupported metadata, truncated payloads, and trailing data. This format is not currently used by the desktop video workbench.

```powershell
.\build\mp48mb.exe encode-block8 input.ppm output.mpb565
.\build\mp48mb.exe decode-block8 output.mpb565 reconstructed.ppm
.\build\mp48mb.exe benchmark-block8 1920 1080
```

At 1920x1080 there are 32,400 blocks, so the RGB565 payload is exactly 64,800 bytes (63.3 KiB); the experimental header makes the total stream 64,820 bytes. This 96:1 payload reduction versus RGB24 is not a quality guarantee: each 8x8 area becomes one flat color.

## 32-bit traversal stack entry

Each pending region is exactly one `uint32_t`:

- Bits 0-11: X origin
- Bits 12-23: Y origin
- Bits 24-27: `log2(block_size)`
- Bits 28-31: flags/reserved

Coordinates and block sizes fit the documented 4096-pixel limit. The traversal uses a preallocated contiguous stack with explicit capacity checks; it does not imply the whole stack fits in one word.

## Tests and benchmarks

```powershell
C:\msys64\ucrt64\bin\mingw32-make.exe test
python -m unittest discover -s src -p "test_*.py"
.\build\mp48mb.exe benchmark 640 480
.\build\mp48mb.exe benchmark 1280 720
.\build\mp48mb.exe benchmark 1920 1080
.\build\mp48mb.exe benchmark-block8 1920 1080
```

The quadtree C++ benchmark performs one warm-up and five measured trials, reporting median single-region versus sequential four-region encode/decode time, output bytes, encode FPS, MSE, and PSNR. `benchmark-block8` measures the same synthetic frame against single- and four-region quadtree variants plus the separate RGB565 8x8 experiment; timing excludes image generation and file I/O and reports median codec encode/decode time after one warm-up and five trials. This synthetic gradient benchmark ran on an AMD Ryzen 5 7600X with MinGW g++ 16.1.0 using `-std=c++20 -O3 -Wall -Wextra -pedantic`; quadtree configuration was max block 256, min block 8, max depth 12, threshold 24:

| Resolution | Mode | Bytes | Encode ms | Decode ms | Encode FPS | MSE | PSNR dB |
|---|---|---:|---:|---:|---:|---:|---:|
| 640x480 | Single | 3,052 | 2.370 | 0.366 | 421.87 | 15.880 | 36.122 |
| 640x480 | Four sequential | 3,288 | 1.990 | 0.341 | 502.46 | 15.501 | 36.227 |
| 1280x720 | Single | 2,452 | 7.222 | 1.134 | 138.46 | 22.637 | 34.583 |
| 1280x720 | Four sequential | 2,488 | 5.974 | 1.034 | 167.38 | 21.843 | 34.738 |
| 1920x1080 | Single | 5,002 | 15.905 | 3.045 | 62.87 | 10.752 | 37.816 |
| 1920x1080 | Four sequential | 5,208 | 13.332 | 2.963 | 75.01 | 10.523 | 37.910 |

These are small synthetic gradient inputs, not representative video-quality or compression claims. Four-region mode used slightly more bytes at the tested larger sizes, with marginally different MSE. The parallel path has not been added; current measurements compare only single-region and sequential four-region modes. The AVX2 block-statistics implementation is runtime-dispatched and retains the scalar reference, but these partition benchmarks use scalar statistics and do not claim SIMD speedup.

## Limitations

- The custom frame codec is a prototype, not an MP4 codec.
- The desktop video path launches the C++ CLI for each frame; this adds process and temporary-file overhead and is not a real-time throughput claim.
- Video audio is omitted; VFR timestamps rely on OpenCV's reported position.
- Four-region processing is sequential and single-worker.
- No FFmpeg/standard-codec MP4 export, audio muxing, temporal prediction in video processing, or HDR pipeline.
- Memory usage is bounded to frame-at-a-time processing but is not currently reported as a live metric.
