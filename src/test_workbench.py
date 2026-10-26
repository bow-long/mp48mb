import csv
import json
import queue
import struct
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from desktop_workbench import BackendServer, VideoWorkbench, frame_mse


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "build" / ("mp48mb.exe" if __import__("os").name == "nt" else "mp48mb")


class WorkbenchTests(unittest.TestCase):
    def test_persistent_backend_process_handles_multiple_frames(self) -> None:
        if not BACKEND.is_file():
            self.skipTest("Build the C++ backend before running the backend protocol test")
        options = {
            "max_block": 64,
            "min_block": 8,
            "max_depth": 12,
            "threshold": 0,
            "regions": "single",
        }
        cancellation = threading.Event()
        frame = np.zeros((16, 24, 3), dtype=np.uint8)
        with BackendServer(BACKEND, cancellation) as server:
            process_id = server.process.pid
            first = server.encode_decode(frame, options)
            second = server.encode_decode(frame, options)
            self.assertEqual(server.process.pid, process_id)
            for encode_ms, decode_ms, payload, reconstructed, profile in (first, second):
                self.assertGreaterEqual(encode_ms, 0.0)
                self.assertGreaterEqual(decode_ms, 0.0)
                self.assertGreater(len(payload), 0)
                self.assertEqual(reconstructed.shape, frame.shape)
                self.assertGreaterEqual(profile["backend_roundtrip_ms"], profile["backend_pipe_write_ms"])

    def test_frame_mse(self) -> None:
        source = np.zeros((3, 4, 3), dtype=np.uint8)
        reconstructed = np.full((3, 4, 3), 2, dtype=np.uint8)
        self.assertEqual(frame_mse(source, reconstructed), 4.0)

    def test_resource_limit_rejects_oversized_dimensions(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured per-axis limit"):
            VideoWorkbench._validate_dimensions(641, 480, {"max_dimension": 640})

    def test_mp4_video_encodes_to_ordered_custom_sequence(self) -> None:
        if not BACKEND.is_file():
            self.skipTest("Build the C++ backend before running the video integration test")

        with tempfile.TemporaryDirectory(prefix="mp48mb-test-") as directory:
            root = Path(directory)
            source = root / "sample.mp4"
            writer = cv2.VideoWriter(
                str(source),
                cv2.VideoWriter_fourcc(*"mp4v"),
                12.0,
                (64, 48),
            )
            if not writer.isOpened():
                self.skipTest("This OpenCV/FFmpeg build has no MP4 video writer")
            for frame_index in range(4):
                frame = np.zeros((48, 64, 3), dtype=np.uint8)
                frame[:, :, 0] = 30 + frame_index * 20
                frame[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :]
                frame[:, :, 2] = np.arange(48, dtype=np.uint8)[:, None]
                writer.write(frame)
            writer.release()

            processor = VideoWorkbench.__new__(VideoWorkbench)
            processor.cancel_event = threading.Event()
            processor.events = queue.Queue(maxsize=2)
            options = {
                "max_block": 64,
                "min_block": 8,
                "max_depth": 12,
                "threshold": 0,
                "max_frames": 10,
                "max_dimension": 4096,
                "regions": "auto",
                "detailed_profiling": True,
            }
            limited_processor = VideoWorkbench.__new__(VideoWorkbench)
            limited_processor.cancel_event = threading.Event()
            limited_processor.events = queue.Queue(maxsize=2)
            limited_options = dict(options)
            limited_options["max_frames"] = 3
            with self.assertRaisesRegex(ValueError, "exceeding the configured limit"):
                limited_processor._process_video(source, root / "limited.mp48seq", BACKEND, limited_options)

            output = root / "sample.mp48seq"
            summary = processor._process_video(source, output, BACKEND, options)
            self.assertEqual(summary["frame_count"], 4)
            self.assertGreater(summary["encoded_bytes"], summary["payload_bytes"])
            self.assertIn(summary["codec_bottleneck"], {"encode", "decode", "similar"})
            self.assertNotIn("preview_original", summary)
            self.assertNotIn("preview_reconstructed", summary)
            self.assertNotIn("preview_ms", summary["profile"])
            for event in list(processor.events.queue):
                self.assertNotIn("preview_original", event)
                self.assertNotIn("preview_reconstructed", event)
            profile_csv = Path(summary["profile_path"])
            self.assertTrue(profile_csv.is_file())
            with profile_csv.open(newline="", encoding="utf-8") as report:
                profile_rows = list(csv.DictReader(report))
            self.assertEqual(len(profile_rows), 4)
            self.assertGreaterEqual(float(profile_rows[0]["codec_encode_ms"]), 0.0)

            with output.open("rb") as sequence:
                self.assertEqual(sequence.read(8), b"MP48SEQ1")
                metadata_size = struct.unpack("<I", sequence.read(4))[0]
                metadata = json.loads(sequence.read(metadata_size))
                self.assertEqual(metadata["width"], 64)
                self.assertEqual(metadata["height"], 48)
                self.assertEqual(metadata["frame_count"], 4)
                self.assertEqual(metadata["pixel_format"], "RGB24")
                self.assertEqual(metadata["audio_policy"], "omitted; this is not an MP4 container")

                timestamps: list[int] = []
                for index in range(4):
                    timestamp_us, payload_size = struct.unpack("<qI", sequence.read(12))
                    payload = sequence.read(payload_size)
                    self.assertEqual(len(payload), payload_size)
                    timestamps.append(timestamp_us)
                    encoded_frame = root / f"frame-{index}.mpf"
                    reconstructed = root / f"frame-{index}.ppm"
                    encoded_frame.write_bytes(payload)
                    subprocess.run(
                        [str(BACKEND), "decode", str(encoded_frame), str(reconstructed)],
                        check=True,
                        capture_output=True,
                    )
                    with reconstructed.open("rb") as ppm:
                        self.assertEqual(ppm.readline().strip(), b"P6")
                        self.assertEqual(ppm.readline().strip(), b"64 48")
                self.assertEqual(sequence.read(), b"")
                self.assertEqual(timestamps, sorted(timestamps))

    def test_video_cancellation_stops_before_first_frame(self) -> None:
        if not BACKEND.is_file():
            self.skipTest("Build the C++ backend before running the video integration test")
        with tempfile.TemporaryDirectory(prefix="mp48mb-cancel-") as directory:
            root = Path(directory)
            source = root / "sample.mp4"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48))
            if not writer.isOpened():
                self.skipTest("This OpenCV/FFmpeg build has no MP4 video writer")
            writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
            writer.release()

            processor = VideoWorkbench.__new__(VideoWorkbench)
            processor.cancel_event = threading.Event()
            processor.cancel_event.set()
            processor.events = queue.Queue(maxsize=2)
            options = {
                "max_block": 64,
                "min_block": 8,
                "max_depth": 12,
                "threshold": 0,
                "max_frames": 10,
                "max_dimension": 4096,
                "regions": "auto",
            }
            output = root / "cancelled.mp48seq"
            with self.assertRaises(InterruptedError):
                processor._process_video(source, output, BACKEND, options)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
