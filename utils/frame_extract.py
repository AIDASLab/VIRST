#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import itertools
import traceback
from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ThreadPool

import av
from PIL import Image
from tqdm import tqdm

# ---------- per-video worker (must be top-level for pickling) ----------

def _compute_out_dir(video_path: Path, src_root: Path, dst_root: Path, preserve_from: str | None) -> Path:
    """
    Build output directory preserving the subpath after `preserve_from` anchor.
    If anchor not found, fallback to src_root-relative structure.
    Example:
      video:  .../videos/academic_source/youcook2/103/8lqdPpg3w08/split_3.mp4
      anchor: videos/academic_source
      out:    .../JPEGImages/youcook2/103/8lqdPpg3w08/split_3/
    """
    if preserve_from:
        anchor_parts = Path(preserve_from).parts
        parts = video_path.parts
        # find the anchor subsequence in the absolute path
        for i in range(0, len(parts) - len(anchor_parts) + 1):
            if tuple(parts[i:i+len(anchor_parts)]) == anchor_parts:
                tail_dirs = parts[i+len(anchor_parts): -1]  # dirs after anchor, exclude filename
                return dst_root.joinpath(*tail_dirs, video_path.stem)

    # fallback: structure relative to src_root (minus filename), then add stem
    rel = video_path.relative_to(src_root)
    return dst_root.joinpath(*rel.parts[:-1], video_path.stem)


def _extract_video_to_folder(args):
    """
    Worker function.
    Returns (video_path_str, frames_written, error_str_or_none)
    """
    (video_path_str, src_root_str, dst_root_str, quality, fps, stride, overwrite, preserve_from) = args
    video_path = Path(video_path_str)
    src_root = Path(src_root_str)
    dst_root = Path(dst_root_str)
    try:
        out_dir = _compute_out_dir(video_path, src_root, dst_root, preserve_from)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Resume support
        if not overwrite:
            existing = sorted(p for p in out_dir.glob("*.jpg") if p.name[:6].isdigit())
            start_index = len(existing)
        else:
            for p in out_dir.glob("*.jpg"):
                try: p.unlink()
                except Exception: pass
            start_index = 0

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        src_fps = float(stream.average_rate) if stream.average_rate else None
        period_s, next_t = (None, None)
        if fps is not None and src_fps:
            period_s = 1.0 / fps
            next_t = 0.0

        frames_written = 0
        idx = 0
        seq = start_index
        for frame in container.decode(video=0):
            pil_img = frame.to_rgb().to_image()

            take = False
            if fps is not None and src_fps:
                t = float(frame.pts * stream.time_base) if frame.pts is not None else (idx / src_fps)
                if t + 1e-9 >= next_t:
                    take = True
                    next_t += period_s
            else:
                if (idx % max(1, stride)) == 0:
                    take = True

            if take:
                seq += 1
                out_path = out_dir / f"{seq:06d}.jpg"
                if overwrite or not out_path.exists():
                    pil_img.save(out_path, format="JPEG", quality=quality, optimize=True)
                frames_written += 1

            idx += 1

        container.close()
        return (video_path_str, frames_written, None)

    except Exception as e:
        import traceback
        return (video_path_str, 0, f"{e}\n{traceback.format_exc()}")

# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from many videos into JPEGImages/<subpath>/<video_stem>/000001.jpg using multiprocessing."
    )
    parser.add_argument("src_root", type=str,
                        help="Root directory searched recursively for video files.")
    parser.add_argument("--dst-root", type=str, default=None,
                        help="Destination root; default: <src_root>/JPEGImages")
    parser.add_argument("--preserve-from", type=str, default="videos/academic_source",
                        help="Anchor path segment. Preserve subpath AFTER this anchor in outputs.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Target FPS (mutually exclusive with --stride).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Take every Nth frame when --fps is not set.")
    parser.add_argument("--quality", type=int, default=90,
                        help="JPEG quality (1-95).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing JPEGs in each output folder.")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help="Number of parallel worker processes.")
    parser.add_argument("--exts", type=str, default=".mp4,.mkv",
                        help="Comma-separated video extensions to include.")
    args = parser.parse_args()

    src_root = Path(args.src_root).resolve()
    # If you pass src_root as dataset root (…/0_30_s_academic_v0_1), this default is correct:
    dst_root = Path(args.dst_root).resolve() if args.dst_root else (src_root / "JPEGImages")
    dst_root.mkdir(parents=True, exist_ok=True)

    exts = tuple(e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                 for e in args.exts.split(","))

    videos = [p for p in src_root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    if not videos:
        print(f"No videos found under {src_root} with extensions {exts}")
        return 1

    print(f"Found {len(videos)} videos. Writing frames under: {dst_root}")
    total_frames = 0
    errors = 0

    job_args = [
        (str(v), str(src_root), str(dst_root), args.quality, args.fps, args.stride, args.overwrite, args.preserve_from)
        for v in videos
    ]

    with Pool(processes=args.workers, maxtasksperchild=50) as pool:
        for video_path_str, n, err in tqdm(
            pool.imap_unordered(_extract_video_to_folder, job_args),
            total=len(job_args), desc="Processing videos", unit="vid"
        ):
            if err:
                errors += 1
                tqdm.write(f"[ERROR] {video_path_str}\n{err}")
            else:
                total_frames += n

    print(f"Done. Total frames extracted: {total_frames}. Videos with errors: {errors}")
    return 0

if __name__ == "__main__":
    main()