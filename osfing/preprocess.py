"""
osfing/preprocess.py — Orchestrate netFound preprocessing for OS fingerprinting.

Converts a PCAP-level split manifest (produced by the NTFM-OSfing pipeline)
into netFound's Arrow format (one directory of .arrow files per split: train /
dev / test).  Internally this wraps ``scripts/preprocess_data.py``, which
drives the full netFound preprocessing chain:

  PCAPs  →  1_filter (C++)  →  2_pcap_splitting (PcapSplitter)
         →  3_field_extraction (C++)  →  Tokenize.py  →  Arrow shards
         →  CollectTokensInFiles.py  →  combined .arrow

Prerequisites (one-time setup):
    cd <netFound_root>/pre_process_src/packets_processing_src
    cmake -B build -DCMAKE_BUILD_TYPE=Release
    cmake --build build
    cp build/1_filter           ../
    cp build/3_field_extraction ../

    # Also requires PcapSplitter (part of PcapPlusPlus) and GNU parallel in PATH.

Input — split_manifest JSON format::

    {
        "train": [{"pcap_path": "/abs/.../traffic.pcap", "label": "Windows 10"}, ...],
        "dev":   [...],
        "test":  [...]
    }

Output::

    <output_dir>/
        train/   ← directory with *.arrow files  (loaded with load_dataset("arrow", data_dir=...))
        dev/
        test/

Usage::

    python osfing/preprocess.py \\
        --split_manifest /path/to/pcap_split_manifest.json \\
        --label_maps     /path/to/label_maps.json \\
        --output_dir     /path/to/arrow_output \\
        --tokenizer_config /path/to/DefaultConfigNoTCPOptions.json \\
        [--work_dir /tmp/netfound_preprocess] \\
        [--keep_work_dir] \\
        [--tcp_options]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split_manifest", required=True,
                   help="JSON: {train: [{pcap_path, label}, ...], dev: [...], test: [...]}")
    p.add_argument("--label_maps", required=True,
                   help="JSON: {label2id: {...}, id2label: {...}, labels_num: N}")
    p.add_argument("--output_dir", required=True,
                   help="Root output directory; Arrow files written to <output_dir>/{train,dev,test}/")
    p.add_argument("--tokenizer_config", required=True,
                   help="Path to netFound tokenizer config JSON (e.g. configs/DefaultConfigNoTCPOptions.json)")
    p.add_argument("--work_dir", default=None,
                   help="Temporary working directory. Defaults to <output_dir>/_work")
    p.add_argument("--keep_work_dir", action="store_true",
                   help="Do not delete the work directory after successful preprocessing.")
    p.add_argument("--tcp_options", action="store_true", default=False,
                   help="Include TCP options in tokenization (requires tcp_options-aware config).")
    p.add_argument("--binaries_dir", default=None,
                   help="Directory containing pre-compiled 1_filter and 3_field_extraction binaries. "
                        "If provided, binaries are copied to pre_process_src/ before preprocessing. "
                        "Use when running from a fresh code snapshot (e.g. labrunner on CESGA).")
    return p


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _get_netfound_root() -> Path:
    """Return the netFound repo root (two levels up: osfing/ → repo root)."""
    return Path(__file__).resolve().parent.parent


def _get_preprocess_script(root: Path) -> Path:
    script = root / "scripts" / "preprocess_data.py"
    if not script.exists():
        raise FileNotFoundError(
            f"netFound preprocess_data.py not found at {script}. "
            "Is the submodule checked out correctly?"
        )
    return script


def _install_binaries(binaries_dir: str, root: Path) -> None:
    """
    Copy pre-compiled binaries from *binaries_dir* into ``pre_process_src/``.

    Used when running from a fresh code snapshot (e.g. labrunner on CESGA)
    where the binaries are stored in a persistent directory outside the snapshot.
    """
    src_dir = Path(binaries_dir)
    dst_dir = root / "pre_process_src"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for binary_name in ("1_filter", "3_field_extraction", "PcapSplitter"):
        src = src_dir / binary_name
        dst = dst_dir / binary_name
        if not src.exists():
            raise FileNotFoundError(
                f"Binary '{binary_name}' not found in binaries_dir={binaries_dir}. "
                "Compile it first — see CLAUDE.md 'netFound one-time setup'."
            )
        shutil.copy2(str(src), str(dst))
        dst.chmod(0o755)
    # Also restore execute permission on the .sh wrapper scripts (stripped by tar extraction)
    # and fix CRLF line endings (git on Windows converts LF→CRLF, breaking the shebang).
    for sh_file in dst_dir.glob("*.sh"):
        sh_file.chmod(0o755)
        content = sh_file.read_bytes()
        if b"\r\n" in content:
            sh_file.write_bytes(content.replace(b"\r\n", b"\n"))
            logger.info("Fixed CRLF line endings in %s", sh_file.name)
    logger.info("Installed binaries from %s into %s", binaries_dir, dst_dir)


def _validate_binaries(root: Path) -> None:
    """Assert that C++ binaries compiled from packets_processing_src/ exist."""
    pre_src = root / "pre_process_src"
    required = [pre_src / "1_filter", pre_src / "3_field_extraction", pre_src / "PcapSplitter"]
    missing = [str(b) for b in required if not b.exists()]
    if missing:
        raise RuntimeError(
            "netFound C++ binaries are missing — compile them first.\n\n"
            "  cd <netFound_root>/pre_process_src/packets_processing_src\n"
            "  cmake -B build -DCMAKE_BUILD_TYPE=Release\n"
            "  cmake --build build\n"
            "  cp build/1_filter           ../\n"
            "  cp build/3_field_extraction ../\n\n"
            "  # PcapSplitter: build from PcapPlusPlus repo root with -DPCAPPP_BUILD_EXAMPLES=ON\n"
            "  # then copy examples_bin/PcapSplitter to pre_process_src/\n\n"
            "Or set binaries_dir in config/osfing/ntfm/netfound.yaml pointing to a\n"
            "persistent directory where all three binaries are stored.\n\n"
            f"Missing binaries: {missing}"
        )


# ---------------------------------------------------------------------------
# PCAP organisation
# ---------------------------------------------------------------------------

def _build_raw_dir(entries: list[dict], raw_dir: Path, label2id: dict) -> int:
    """
    Populate ``raw_dir/<label_name>/`` with symlinks to each assigned PCAP.

    The symlink name is derived from the PCAP's parent directory (the timestamp
    folder) so it is unique within the class directory.

    Returns the number of PCAPs successfully linked.
    """
    count = 0
    for entry in entries:
        pcap_path = Path(entry["pcap_path"]).resolve()
        label_name: str = entry["label"]

        if label_name not in label2id:
            logger.warning("Label %r not in label_maps — skipping %s", label_name, pcap_path)
            continue
        if not pcap_path.exists():
            logger.warning("PCAP not found, skipping: %s", pcap_path)
            continue

        class_dir = raw_dir / label_name
        class_dir.mkdir(parents=True, exist_ok=True)

        # Use the timestamp directory name as a unique file stem.
        link_name = class_dir / (pcap_path.parent.name + ".pcap")
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        os.symlink(str(pcap_path), str(link_name))
        count += 1

    return count


# ---------------------------------------------------------------------------
# Preprocessing runner
# ---------------------------------------------------------------------------

def _run_preprocess_for_split(
    split_name: str,
    split_dir: Path,
    preprocess_script: Path,
    tokenizer_config: str,
    tcp_options: bool,
    keep_intermediates: bool = False,
) -> None:
    """Invoke ``scripts/preprocess_data.py --action finetune --combined`` for one split."""
    cmd = [
        sys.executable,
        str(preprocess_script),
        "--input_folder", str(split_dir),
        "--action", "finetune",
        "--tokenizer_config", tokenizer_config,
        "--combined",
    ]
    if tcp_options:
        cmd.append("--tcp_options")
    if keep_intermediates:
        cmd.append("--keep_intermediates")

    logger.info("[%s] Running: %s", split_name, " ".join(cmd))
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)

    for line in (result.stdout or "").splitlines():
        logger.info("[%s] %s", split_name, line)
    for line in (result.stderr or "").splitlines():
        logger.warning("[%s stderr] %s", split_name, line)

    if result.returncode != 0:
        raise RuntimeError(
            f"netFound preprocess_data.py failed for split '{split_name}' "
            f"(exit code {result.returncode}). See log above for details."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _make_arg_parser().parse_args()

    netfound_root = _get_netfound_root()
    preprocess_script = _get_preprocess_script(netfound_root)
    if args.binaries_dir:
        _install_binaries(args.binaries_dir, netfound_root)
    _validate_binaries(netfound_root)

    # Load manifests
    with open(args.split_manifest, encoding="utf-8") as fh:
        split_manifest: dict = json.load(fh)
    with open(args.label_maps, encoding="utf-8") as fh:
        label_maps: dict = json.load(fh)
    label2id: dict = label_maps["label2id"]

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / "_work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev", "test"):
        entries: list[dict] = split_manifest.get(split, [])
        if not entries:
            logger.warning("Split '%s' has no entries in split_manifest — skipping.", split)
            continue

        split_work = work_dir / split
        split_work.mkdir(parents=True, exist_ok=True)

        logger.info("=== Split: %s — organising %d PCAPs ===", split, len(entries))
        n_linked = _build_raw_dir(entries, raw_dir=split_work / "raw", label2id=label2id)
        if n_linked == 0:
            logger.warning("No PCAPs linked for split '%s' — skipping.", split)
            continue
        logger.info("  Linked %d PCAPs", n_linked)

        _run_preprocess_for_split(
            split_name=split,
            split_dir=split_work,
            preprocess_script=preprocess_script,
            tokenizer_config=args.tokenizer_config,
            tcp_options=args.tcp_options,
            keep_intermediates=args.keep_work_dir,
        )

        # Move combined Arrow files to output_dir/<split>/
        combined_src = split_work / "final" / "combined"
        if not combined_src.exists():
            raise RuntimeError(
                f"Expected combined Arrow directory not found after preprocessing: {combined_src}. "
                "Did preprocess_data.py succeed?"
            )

        combined_dst = output_dir / split
        combined_dst.mkdir(parents=True, exist_ok=True)
        arrow_files = list(combined_src.glob("*.arrow"))
        if not arrow_files:
            raise RuntimeError(f"No .arrow files found in {combined_src}.")

        for arrow_file in arrow_files:
            dst = combined_dst / arrow_file.name
            if dst.exists():
                dst.unlink()
            shutil.move(str(arrow_file), str(dst))

        n_arrows = len(list(combined_dst.glob("*.arrow")))
        logger.info("  Wrote %d Arrow files to %s", n_arrows, combined_dst)

        if not args.keep_work_dir:
            # This split's Arrow output is safely in output_dir; drop the split's
            # work tree now instead of at the end so at most one split's
            # intermediates exist at a time (inode-quota hygiene on Lustre).
            shutil.rmtree(split_work, ignore_errors=True)
            logger.info("  Cleaned split work directory: %s", split_work)

    if not args.keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("Cleaned up work directory: %s", work_dir)

    logger.info("Preprocessing complete. Arrow files in %s/{train,dev,test}/", output_dir)


if __name__ == "__main__":
    main()
