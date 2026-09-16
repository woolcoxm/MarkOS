# MarkOS — build pipeline walkthrough

`os/build.sh` turns this repository into flashable appliance images with zero
manual steps. Run it on a Linux builder (native or WSL2).

## What you need (host)

- Build toolchain: `make gcc g++ patch perl tar unzip cpio rsync bc wget file cmake`
- Git
- Rust (rustup) with `aarch64-unknown-linux-gnu` target (build.sh adds it)
- ~15 GB free disk, network access for the first fetch (pinned Buildroot,
  rpi kernel tarball, firmware blobs, crates)

## Build

```sh
cd os
./build.sh --variant both        # or: sd / ssd
```

Artifacts in `os/output/`:

| file | purpose |
|---|---|
| `markos-sd.img` (+ `.sha256`) | SD-card variant: squashfs read-only root |
| `markos-ssd.img` (+ `.sha256`) | USB SSD / NVMe variant: writable ext4 root |

Both share the same layout: `p1` FAT32 boot · `p2` rootfsA · `p3` rootfsB
(A/B slots) · `p4` data (models/logs/state; grows to fill the media on
first boot).

## How it works

1. `build.sh` clones the **pinned Buildroot release** (2025.02) and registers
   this repo's `os/` directory as a `BR2_EXTERNAL` tree.
2. `os/configs/markos_pi5_defconfig` is applied. It mirrors upstream's
   `raspberrypi5_defconfig` (kernel rpi-6.6 `bcm2712` defconfig, Pi firmware,
   genimage) plus the appliance delta: runit supervision, dropbear (no
   passwords), WPA3-capable wpa_supplicant, avahi, nftables, resize2fs/sfdisk,
   libgpiod, curl, and the `markos-engine` package.
3. The **engine** is built by Buildroot's cross toolchain from the local
   `engine/` workspace with `--no-default-features --features llama,tls` and
   `-C target-cpu=cortex-a76`. The cargo `llama` feature compiles
   llama.cpp/ggml through cmake against the same toolchain — this is the one
   deliberately reused layer (see design doc §7); everything above ggml is
   MarkOS.
4. `board/markos/overlay/` supplies the runtime: busybox init + `s6-svscan`
   service supervision (`markos-engine`, `avahi`, `watchdogd`, `boot-confirm`,
   conditional `dropbear`), the S01–S05 boot stages (data mounts, writable-/etc
   strategy for squashfs, network + IPv4LL rescue address, nftables firewall,
   factory-reset button check, first-boot data-partition growth), and the
   `markos-update` / `markos-factory-reset` tools.
5. `post-image.sh` produces both A/B rootfs slots and runs `genimage` to emit
   the final image + sha256.

## Boot-testing in QEMU (no hardware needed)

The kernel fragment enables virtio-mmio (`CONFIG_VIRTIO_{MMIO,BLK,NET}`) so the
image boots on `qemu-system-aarch64 -M virt` — the same validation the CI story
uses (harmless on real Pi hardware, which has no virtio devices):

```sh
# 1. configure an image with the installer (provision file into boot FAT)
markos-installer --config qemu-install.toml --image markos-sd.img                  --target sd --out qemu-test.img

# 2. (optional) hardcode a model: grow p4, seed /data/state/markos.json + model
truncate -s 5G qemu-test.img
echo ",+" | sfdisk --no-reread -N 4 qemu-test.img
LOOP=$(losetup -Pf --show qemu-test.img); partx -u $LOOP; resize2fs ${LOOP}p4
mount ${LOOP}p4 /mnt   # → /mnt/state/markos.json with auto_load, /mnt/state/models/

# 3. boot: kernel direct, appliance disk attached, ports forwarded
qemu-system-aarch64 -M virt -cpu cortex-a76 -smp 4 -m 3072   -kernel <build-sd>/images/Image   -append "root=PARTUUID=6d61726b-02 rootwait console=ttyAMA0"   -drive if=none,file=qemu-test.img,format=raw,id=hd0   -device virtio-blk-device,drive=hd0   -netdev user,id=n0,hostfwd=tcp::8080-:8080,hostfwd=tcp::8081-:80   -device virtio-net-device,netdev=n0   -display none -serial file:serial.log

# 4. prove it: healthz → provisioned login → model resident → completion
curl localhost:8080/healthz
curl -c ck -X POST localhost:8081/api/login -d '{"username":"admin","password":"..."}'
curl localhost:8080/v1/models           # resident:true after auto-load
curl localhost:8080/v1/chat/completions -d '{"model":"...","messages":[...]}'
```

This exact flow is what validated the appliance in this repo: booted headless,
s6-supervised engine up in ~35 s, installer-provisioned admin login, hardcoded
Qwen2.5-0.5B Q4_K_M auto-loaded and served over the OpenAI-compatible API.
(TCG emulation is slow — generation takes ~100 s for a short answer; on real
A76 silicon the same path is interactive-speed.)

## Verify on hardware

1. Flash with the installer (`markos-installer --config … --write …`) or
   `dd`; the installer's config injection is what makes the box reachable.
2. First boot: watchdog daemon waits out the data-partition growth, the
   engine consumes `/boot/markos/provision.env` (creates the admin from the
   installer's argon2id hash, starts the model preseed), then renames it to
   `provision.done`.
3. Check: `http://<mdns-name>.local` (web UI), `http://<host>:8080/healthz`
   (API), `http://169.254.9.1:4444` with a direct Ethernet cable (rescue).

## Boot-order notes for USB/NVMe

The Pi 5 firmware must be told to try USB/NVMe. One-time on the board:
boot any OS from SD once and run `raspi-config` → *Boot Order* (or
`rpi-eeprom-config`, set `BOOT_ORDER=0xf41n` for NVMe-first, `0xf41u` for
USB-first). The installer prints this reminder when you pick SSD/NVMe.
The boot partition itself is identical across all three media.

## Updating

Manual, A/B, rollback-safe: put a new `rootfs.squashfs|ext4` + sha256 where
the box can fetch it, then `markos-update <image> <sha256>` (or the web UI's
update flow once a manifest URL is configured). The box reboots into the
inactive slot via firmware `tryboot`; `boot-confirm` commits only after the
engine proves healthy for 3 minutes — otherwise the next boot is the old
slot, untouched. Nothing ever auto-installs.
