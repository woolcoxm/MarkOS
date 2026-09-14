#!/usr/bin/env python3
"""MarkOS installer (Pi-6): bake an appliance SD image.

Builds a bootable MBR + FAT32 image holding the kernel, the GGUF model,
and MARKOS.CFG — the appliance's only install-time configuration surface
(network identity, control port, admin token). The kernel reads MARKOS.CFG
once at boot; nothing else configures it outside LLM controls.

Example:
  python3 scripts/install.py --image sd/markos.img \
      --kernel kernel_2712.img --model MODEL.BIN \
      --ip 192.168.1.50 --port 8080 --token "$(head -c 16 /dev/urandom | xxd -p)"

Requires: sfdisk, mkfs.vfat, mcopy (mtools) — same toolchain the test
image rules use. `--firmware-dir` optionally copies Raspberry Pi 5
firmware blobs (start4.elf, fixup4.dat, config.txt) for real hardware
boots; the QEMU acceptance gate does not need them."""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

BOOT_PART_START = 2048  # sectors: 1 MiB-aligned boot partition


def run(cmd, **kw):
    subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, help="output image path")
    ap.add_argument("--kernel", required=True, help="kernel image (kernel_2712.img)")
    ap.add_argument("--model", required=True, help="GGUF model file (MODEL.BIN)")
    ap.add_argument("--ip", default="10.0.2.15", help="appliance IPv4")
    ap.add_argument("--port", type=int, default=8080, help="TCP control port")
    ap.add_argument("--token", default="", help="admin token (empty = none)")
    ap.add_argument("--size-mb", type=int, default=64, help="image size in MiB")
    ap.add_argument("--firmware-dir", default=None,
                    help="optional dir with Pi firmware files to copy")
    args = ap.parse_args()

    if os.path.exists(args.image):
        os.remove(args.image)
    run(["dd", "if=/dev/zero", f"of={args.image}", "bs=1M",
         f"count={args.size_mb}", "status=none"])

    # One FAT32 partition spanning the rest of the disk (MBR).
    run(["sfdisk", args.image], input=f"{BOOT_PART_START},,0x0c\n",
        text=True, stdout=subprocess.DEVNULL)

    part_off = BOOT_PART_START * 512
    with tempfile.NamedTemporaryFile(suffix=".img", delete=False) as tmp:
        part = tmp.name
    size_mb = (os.path.getsize(args.image) - part_off) // (1024 * 1024)
    run(["dd", f"if={args.image}", f"of={part}", "bs=512",
         f"skip={BOOT_PART_START}", "status=none"])
    run(["mkfs.vfat", "-F32", part], stdout=subprocess.DEVNULL)

    mcopy = lambda src, dst: run(["mcopy", "-i", part, src, dst])
    mcopy(args.kernel, "::/KERNEL_2712.IMG")
    mcopy(args.model, "::/MODEL.BIN")

    cfg = ("# MarkOS appliance configuration (baked at install time)\n"
           f"ip={args.ip}\n"
           f"port={args.port}\n"
           f"token={args.token}\n")
    with tempfile.NamedTemporaryFile("w", suffix=".CFG", delete=False) as tmp:
        tmp.write(cfg)
        cfg_path = tmp.name
    mcopy(cfg_path, "::/MARKOS.CFG")

    if args.firmware_dir:
        for name in sorted(os.listdir(args.firmware_dir)):
            mcopy(os.path.join(args.firmware_dir, name), f"::/{name}")

    run(["dd", f"if={part}", f"of={args.image}", "bs=512",
         f"seek={BOOT_PART_START}", "conv=notrunc", "status=none"])
    os.remove(part)
    os.remove(cfg_path)

    print(f"install: {args.image} — kernel={os.path.basename(args.kernel)} "
          f"model={os.path.basename(args.model)} ip={args.ip} "
          f"port={args.port} token={'set' if args.token else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
