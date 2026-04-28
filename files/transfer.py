#!/usr/bin/env python3
"""
transfer.py — Quick file transfer to USB pendrive after landing.

Plug in the pendrive AFTER the drone has landed, then run:

    python3 transfer.py landing_20250428_143022.mp4
    python3 transfer.py landing_*.mp4 landing_*.log
    python3 transfer.py *.mp4 *.log *.csv
    python3 transfer.py --all-videos
    python3 transfer.py --all-logs
    python3 transfer.py --all

The script will:
  1. Auto-detect the USB pendrive (wait up to 30s if not yet mounted)
  2. Create a timestamped folder on the drive: X_LANDING_<date>/
  3. Copy files with progress
  4. Verify copied sizes match
  5. Safely sync & unmount the drive so you can pull it out

No need to know which /dev/sdX or mount point — it figures it out.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime


# ── Where to look for USB mounts on Jetson (Ubuntu) ──────────
MEDIA_DIRS = [
    "/media",           # Ubuntu auto-mount: /media/<user>/<label>
    "/mnt",             # manual mounts
    "/run/media",       # some distros
]

HOME = os.path.expanduser("~")

# File patterns for --all shortcuts
VIDEO_PATTERNS = ["*.mp4", "*.avi", "*.mkv"]
LOG_PATTERNS   = ["*.log"]
CSV_PATTERNS   = ["*.csv"]


def fmt_size(b):
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def find_usb_mount():
    """
    Find a mounted USB drive. Checks lsblk for removable block devices,
    then returns the mount point.
    """
    try:
        out = subprocess.check_output(
            ["lsblk", "-rno", "NAME,MOUNTPOINT,RM,TYPE"],
            text=True
        )
    except Exception:
        return None

    for line in out.strip().split("\n"):
        parts = line.split()
        if len(parts) < 4:
            continue
        name, mount, removable, dtype = parts[0], parts[1], parts[2], parts[3]
        # RM=1 means removable, TYPE=part means it's a partition
        if removable == "1" and dtype == "part" and mount and mount != "":
            return mount

    return None


def find_usb_from_media():
    """
    Fallback: scan /media/<user>/ for any mounted directory that isn't
    the root filesystem.
    """
    for base in MEDIA_DIRS:
        if not os.path.isdir(base):
            continue
        # /media/<user>/<drive_label>
        for user_dir in os.listdir(base):
            user_path = os.path.join(base, user_dir)
            if os.path.isdir(user_path):
                for drive in os.listdir(user_path):
                    drive_path = os.path.join(user_path, drive)
                    if os.path.ismount(drive_path) or os.path.isdir(drive_path):
                        # Quick write test
                        try:
                            test = os.path.join(drive_path, ".transfer_test")
                            with open(test, "w") as f:
                                f.write("ok")
                            os.remove(test)
                            return drive_path
                        except (PermissionError, OSError):
                            continue
    return None


def wait_for_usb(timeout=30):
    """Wait for a USB drive to appear, polling every 2s."""
    print("[USB] Looking for USB drive …")
    t0 = time.time()
    while time.time() - t0 < timeout:
        mount = find_usb_mount()
        if mount:
            return mount
        mount = find_usb_from_media()
        if mount:
            return mount
        remaining = timeout - (time.time() - t0)
        print(f"\r[USB] No USB drive found — waiting … ({remaining:.0f}s left)  ",
              end="", flush=True)
        time.sleep(2)
    print()
    return None


def resolve_files(patterns):
    """Expand glob patterns from the home directory."""
    files = []
    for pattern in patterns:
        # If it's an absolute path or already has a slash, glob as-is
        if os.path.sep in pattern or os.path.isabs(pattern):
            matches = sorted(glob.glob(pattern))
        else:
            # Glob relative to home dir (where flight scripts run)
            matches = sorted(glob.glob(os.path.join(HOME, pattern)))
            # Also try current directory
            matches += sorted(glob.glob(pattern))
        files.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        real = os.path.realpath(f)
        if real not in seen and os.path.isfile(real):
            seen.add(real)
            unique.append(real)
    return unique


def copy_with_progress(src, dst):
    """Copy file with a simple progress indicator."""
    size = os.path.getsize(src)
    copied = 0
    chunk = 4 * 1024 * 1024  # 4 MB chunks

    with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
        while True:
            buf = fsrc.read(chunk)
            if not buf:
                break
            fdst.write(buf)
            copied += len(buf)
            pct = copied / size * 100 if size > 0 else 100
            bar_len = 30
            filled = int(bar_len * copied / size) if size > 0 else bar_len
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r    {bar} {pct:5.1f}%  {fmt_size(copied)}/{fmt_size(size)}  ",
                  end="", flush=True)
    print()


def safe_eject(mount_point):
    """Sync and attempt to unmount the drive safely."""
    print("\n[USB] Syncing …")
    subprocess.run(["sync"], check=False)

    print("[USB] Unmounting …")
    result = subprocess.run(["umount", mount_point],
                            capture_output=True, text=True)
    if result.returncode == 0:
        print("[USB] ✓ Drive safely unmounted — you can pull it out now!")
    else:
        # Try with sudo
        result2 = subprocess.run(["sudo", "umount", mount_point],
                                 capture_output=True, text=True)
        if result2.returncode == 0:
            print("[USB] ✓ Drive safely unmounted — you can pull it out now!")
        else:
            print("[USB] ⚠ Could not unmount automatically.")
            print("      Run: sudo umount " + mount_point)
            print("      Or just run 'sync' and wait 5s before pulling the drive.")


def main():
    p = argparse.ArgumentParser(
        description="Transfer flight files to USB pendrive",
        epilog="Examples:\n"
               "  python3 transfer.py landing_20250428.mp4\n"
               "  python3 transfer.py *.mp4 *.log\n"
               "  python3 transfer.py --all-videos\n"
               "  python3 transfer.py --all",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("files", nargs="*", help="Files or glob patterns to transfer")
    p.add_argument("--all-videos", action="store_true",
                   help="Transfer all .mp4/.avi/.mkv in ~/")
    p.add_argument("--all-logs", action="store_true",
                   help="Transfer all .log files in ~/")
    p.add_argument("--all-csv", action="store_true",
                   help="Transfer all .csv files in ~/")
    p.add_argument("--all", action="store_true",
                   help="Transfer all videos, logs, and CSVs")
    p.add_argument("--no-eject", action="store_true",
                   help="Don't unmount the drive after transfer")
    p.add_argument("--timeout", type=int, default=30,
                   help="Seconds to wait for USB drive (default 30)")
    args = p.parse_args()

    # ── Build file list ───────────────────────────────────────
    patterns = list(args.files) if args.files else []

    if args.all or args.all_videos:
        patterns.extend(VIDEO_PATTERNS)
    if args.all or args.all_logs:
        patterns.extend(LOG_PATTERNS)
    if args.all or args.all_csv:
        patterns.extend(CSV_PATTERNS)

    if not patterns:
        p.print_help()
        print("\n[!] No files specified. Use filenames, globs, or --all.")
        sys.exit(1)

    files = resolve_files(patterns)
    if not files:
        print(f"[!] No files matched: {' '.join(patterns)}")
        sys.exit(1)

    total_size = sum(os.path.getsize(f) for f in files)
    print(f"\n[FILES] {len(files)} file(s) to transfer ({fmt_size(total_size)}):")
    for f in files:
        print(f"    {os.path.basename(f):40s}  {fmt_size(os.path.getsize(f))}")

    # ── Find USB drive ────────────────────────────────────────
    mount = wait_for_usb(timeout=args.timeout)
    if not mount:
        print("[!] ERROR: No USB drive found. Is it plugged in?")
        print("    Tip: try 'lsblk' to check, or 'sudo mount /dev/sda1 /mnt'")
        sys.exit(1)

    # Check free space
    stat = os.statvfs(mount)
    free = stat.f_bavail * stat.f_frsize
    print(f"\n[USB] Found drive: {mount}")
    print(f"[USB] Free space:  {fmt_size(free)}")

    if total_size > free:
        print(f"[!] ERROR: Not enough space! Need {fmt_size(total_size)}, "
              f"have {fmt_size(free)}")
        sys.exit(1)

    # ── Create destination folder ─────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = os.path.join(mount, f"X_LANDING_{ts}")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"[USB] Destination: {dest_dir}\n")

    # ── Copy files ────────────────────────────────────────────
    copied_ok = 0
    t0 = time.time()

    for i, src in enumerate(files, 1):
        name = os.path.basename(src)
        dst = os.path.join(dest_dir, name)
        src_size = os.path.getsize(src)

        print(f"  [{i}/{len(files)}] {name}  ({fmt_size(src_size)})")
        copy_with_progress(src, dst)

        # Verify
        dst_size = os.path.getsize(dst)
        if dst_size == src_size:
            print(f"         ✓ verified")
            copied_ok += 1
        else:
            print(f"         ✗ SIZE MISMATCH: src={src_size} dst={dst_size}")

    elapsed = time.time() - t0
    speed = total_size / elapsed if elapsed > 0 else 0

    print(f"\n{'='*55}")
    print(f"  ✓ {copied_ok}/{len(files)} files transferred")
    print(f"  {fmt_size(total_size)} in {elapsed:.1f}s ({fmt_size(speed)}/s)")
    print(f"  → {dest_dir}")
    print(f"{'='*55}")

    # ── Eject ─────────────────────────────────────────────────
    if not args.no_eject:
        safe_eject(mount)
    else:
        subprocess.run(["sync"], check=False)
        print("\n[USB] Synced. Drive still mounted (--no-eject).")


if __name__ == "__main__":
    main()
