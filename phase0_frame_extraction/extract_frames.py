#!/usr/bin/env python3
"""
Phase 0: turn a video into PNG frames sampled at a chosen frame rate.

Uses ffmpeg / ffprobe (must be on PATH, or set FFMPEG_BIN / FFPROBE_BIN).
Frames are written as ``frame_000001.png`` so the rest of the pipeline's
``frame_<n>_fish_<k>`` naming keeps working.

    python extract_frames.py --video path/to/clip.mp4 --fps 5
    python extract_frames.py --video clip.mp4 --probe        # just print video info

The interactive GUI (run_pipeline_gui.py) calls the helpers here directly.
"""

import sys
import json
import shutil
import argparse
import subprocess
from collections import deque
from pathlib import Path

for _p in Path(__file__).resolve().parents:
    if (_p / "config.py").exists():
        sys.path.insert(0, str(_p))
        break
import config


class FFmpegNotFound(RuntimeError):
    pass


def _require_tool(binary: str) -> str:
    resolved = shutil.which(binary)
    if resolved is None:
        raise FFmpegNotFound(
            f"'{binary}' was not found on PATH. Install ffmpeg "
            f"(https://ffmpeg.org) or set FFMPEG_BIN / FFPROBE_BIN."
        )
    return resolved


TS_EXTENSIONS = {".mts", ".m2ts", ".ts"}


def _ts_input_args(video_path: Path) -> list:
    """
    Extra ffmpeg/ffprobe input args for MPEG-TS camcorder files.

    AVCHD .mts/.m2ts often drop out of sync mid-stream; the default
    resync_size (64 KB) makes ffmpeg give up and hang. Raising it lets the
    demuxer scan a few seconds ahead and recover. No effect on healthy files
    and none at all on non-TS containers.
    """
    if Path(video_path).suffix.lower() in TS_EXTENSIONS:
        return ["-resync_size", "50000000"]
    return []


def _parse_rate(rate_text: str) -> float:
    """ffprobe returns frame rates as fractions like '30000/1001'."""
    rate_text = (rate_text or "").strip()
    if not rate_text or rate_text == "0/0":
        return 0.0
    if "/" in rate_text:
        num, den = rate_text.split("/", 1)
        den = float(den)
        return float(num) / den if den else 0.0
    return float(rate_text)


def probe_video(video_path) -> dict:
    """Return {fps, duration, frame_count, width, height} for a video file."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    ffprobe = _require_tool(config.FFPROBE_BIN)
    ts_args = _ts_input_args(video_path)
    result = subprocess.run(
        [
            ffprobe, "-v", "error", *ts_args,
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height,duration",
            "-show_entries", "format=duration",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed:\n{result.stderr.strip()}")

    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}

    avg_rate = _parse_rate(stream.get("avg_frame_rate"))
    r_rate = _parse_rate(stream.get("r_frame_rate"))
    fps = avg_rate or r_rate                        # frames per second

    duration = 0.0
    for source in (stream.get("duration"), fmt.get("duration")):
        try:
            duration = float(source)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = int(round(fps * duration)) if fps and duration else 0

    # AVCHD / MPEG-TS (.mts, .m2ts, .ts) usually carry no duration or frame
    # count in the header. Fall back to counting video packets (no full decode).
    # For interlaced 1080i the packet count tracks r_frame_rate (the field
    # rate), so divide by that to get seconds, then multiply back by fps.
    if (duration <= 0 or frame_count <= 0) and fps > 0:
        config.emit_step("counting frames (container has no duration header)")
        counted = _count_video_packets(ffprobe, video_path, ts_args)
        if counted > 0:
            duration = counted / (r_rate or fps)
            frame_count = int(round(duration * fps))

    return {
        "fps": round(fps, 4),
        "duration": round(duration, 3),
        "frame_count": frame_count,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def _count_video_packets(ffprobe: str, video_path: Path, ts_args=()) -> int:
    """Number of packets in the first video stream (scans the file, no decode)."""
    result = subprocess.run(
        [
            ffprobe, "-v", "error", *ts_args,
            "-select_streams", "v:0",
            "-count_packets",
            "-show_entries", "stream=nb_read_packets",
            "-of", "json",
            str(video_path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return 0
    try:
        stream = (json.loads(result.stdout).get("streams") or [{}])[0]
        return int(stream.get("nb_read_packets") or 0)
    except (ValueError, KeyError):
        return 0


def estimate_extracted_frames(duration: float, sample_fps: float) -> int:
    """How many frames ffmpeg will write for ``-vf fps=sample_fps``."""
    if duration <= 0 or sample_fps <= 0:
        return 0
    return max(1, int(duration * sample_fps))


def suggest_sample_fps(video_fps: float, duration: float) -> list[dict]:
    """
    Offer a few sensible sampling rates <= the video's own fps, each with an
    estimated extracted-frame count. Marks one 'recommended'.
    """
    video_fps = float(video_fps or 0)
    candidates = {
        fps for fps in config.FPS_SUGGESTIONS
        if fps <= video_fps + 1e-6
    }
    if video_fps > 0:
        candidates.add(round(video_fps, 2))          # full rate
        candidates.add(round(video_fps / 2, 2))      # half rate

    # Recommend the rate that extracts ~TARGET_FRAMES frames (a frame holds
    # ~6 fish, so ~3000 frames -> ~18000 ROI crops). Fall back to a fixed fps
    # when the duration is unknown.
    if duration > 0:
        ideal = config.TARGET_FRAMES / duration
        ideal = min(ideal, video_fps) if video_fps > 0 else ideal
        ideal = round(ideal, 1 if ideal >= 1 else 2)
        if ideal > 0:
            candidates.add(ideal)
        recommended = min(candidates, key=lambda c: abs(c - ideal))
    else:
        recommended = min(
            [c for c in candidates if c >= config.DEFAULT_SAMPLE_FPS]
            or [max(candidates or [video_fps])]
        )

    rows = []
    for fps in sorted(c for c in candidates if c > 0):
        rows.append({
            "fps": fps,
            "estimated_frames": estimate_extracted_frames(duration, fps),
            "recommended": abs(fps - recommended) < 1e-6,
        })
    return rows


def format_probe_report(video_path, info: dict, suggestions: list[dict]) -> str:
    lines = [
        f"Video:      {Path(video_path).name}",
        f"Resolution: {info['width']} x {info['height']}",
        f"Duration:   {info['duration']:.1f} s",
        f"Frame rate: {info['fps']:.2f} fps",
        f"Frames:     {info['frame_count']}",
        "",
        "Suggested sampling rates (frames actually sent through detection + ViT):",
    ]
    for row in suggestions:
        tag = "  <-- recommended" if row["recommended"] else ""
        lines.append(
            f"  {row['fps']:>6.2f} fps  ->  ~{row['estimated_frames']:>7d} frames{tag}"
        )
    lines.append("")
    lines.append(
        f"Recommended rate targets ~{config.TARGET_FRAMES} frames "
        f"(~6 fish/frame -> ~{config.TARGET_FRAMES * 6} ROI crops). Every frame "
        f"goes through the detector; each detected fish becomes one crop for the ViT."
    )
    return "\n".join(lines)


def extract_frames(video_path, sample_fps: float, output_dir=None, clean: bool = True,
                   expected_frames: int = 0) -> int:
    """Write PNG frames and return how many were produced."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")

    output_dir = Path(output_dir) if output_dir else config.FRAMES_DIR

    config.emit_step("preparing output folder")
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg = _require_tool(config.FFMPEG_BIN)
    pattern = str(output_dir / "frame_%06d.png")

    if not expected_frames:
        info = probe_video(video_path)
        expected_frames = estimate_extracted_frames(info["duration"], sample_fps)

    # _ts_input_args : raise resync_size for AVCHD .mts/.m2ts that desync
    #                  mid-stream (otherwise ffmpeg hangs); no-op for mp4/mov
    # -fflags +genpts : rebuild sane timestamps
    # -map 0:v:0 -an  : first video stream only, ignore audio
    # stderr is merged into stdout and drained by one loop, so a flood of MTS
    # decoder warnings can't fill a pipe buffer and deadlock ffmpeg.
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        *_ts_input_args(video_path),
        "-fflags", "+genpts",
        "-i", str(video_path),
        "-map", "0:v:0", "-an",
        "-vf", f"fps={sample_fps}",
        "-progress", "pipe:1", "-nostats",
        pattern,
    ]
    if expected_frames > 0:
        config.emit_step(f"extracting ~{expected_frames} frames with ffmpeg")
    else:
        config.emit_step("extracting frames with ffmpeg (count unknown)")
    config.emit_progress(0, expected_frames or 1, "extracting")

    progress_keys = ("frame=", "fps=", "stream_", "bitrate=", "total_size=",
                     "out_time", "dup_frames", "drop_frames", "speed=", "progress=")
    err_tail = deque(maxlen=25)
    warn_count = 0

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    for raw in proc.stdout:
        line = raw.strip()
        if line.startswith("frame="):
            try:
                done = int(line.split("=", 1)[1])
                config.emit_progress(min(done, expected_frames or done),
                                     expected_frames or done, "extracting")
            except ValueError:
                pass
        elif line and not line.startswith(progress_keys):
            warn_count += 1
            err_tail.append(line)

    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed:\n" + "\n".join(err_tail))
    if warn_count:
        print(f"(ffmpeg reported {warn_count} decoder messages — usually harmless for AVCHD)")

    produced = len(list(output_dir.glob("frame_*.png")))
    config.emit_progress(produced, produced, "extracting")
    print(f"Extracted {produced} frames to {output_dir}")
    return produced


def _cli():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video file")
    parser.add_argument("--fps", type=float, default=None,
                        help="Sampling frame rate (default: interactive prompt)")
    parser.add_argument("--output", type=Path, default=config.FRAMES_DIR,
                        help="Folder to write PNG frames into")
    parser.add_argument("--probe", action="store_true",
                        help="Only print video info + fps suggestions, do not extract")
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"Video file does not exist: {args.video}")

    config.emit_step("probing video")
    info = probe_video(args.video)
    suggestions = suggest_sample_fps(info["fps"], info["duration"])
    print(format_probe_report(args.video, info, suggestions))

    if args.probe:
        return

    sample_fps = args.fps
    if sample_fps is None:
        default = next((r["fps"] for r in suggestions if r["recommended"]),
                       config.DEFAULT_SAMPLE_FPS)
        answer = input(f"\nSampling fps to use [{default}]: ").strip()
        sample_fps = float(answer) if answer else default

    if sample_fps > info["fps"] + 1e-6:
        print(f"\nRequested {sample_fps} fps > video fps {info['fps']:.2f}; "
              f"clamping to {info['fps']:.2f}.")
        sample_fps = info["fps"]

    est = estimate_extracted_frames(info["duration"], sample_fps)
    est_text = f"~{est} frames" if est else "an unknown number of frames"
    print(f"\nAbout to extract {est_text} at {sample_fps} fps.")
    extract_frames(args.video, sample_fps, args.output, expected_frames=est)


if __name__ == "__main__":
    _cli()
