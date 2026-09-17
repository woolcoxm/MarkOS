# Hardware bring-up debug log — 2026-09-16 (axcl image on d0-stepping Pi 5)

Working notes from tonight's flash/boot session. Current state at the bottom.

## What the boot-debug flight recorder proved

The new image's `S00markos-debug` script (runs at first userspace breath)
wrote its record to the boot FAT on the failing boots:

```
==== markos boot Thu Jan  1 00:00:15 UTC 1970 ====
-- S00 kernel partitions --
179 0 999874560 mmcblk0          # the 1 TB card, full geometry visible
179 1 524288 mmcblk0p1           # boot FAT
179 2 73524 mmcblk0p2            # slot A (squashfs)
179 3 73524 mmcblk0p3            # slot B
179 4 524288 mmcblk0p4           # data (fresh, needs first-boot grow)
-- S00 cmdline -- (complete, with module_blacklist=...)
```

**Conclusion: the kernel boots and starts userspace.** Bootloader, firmware,
FAT, kernel, and partition table are all exonerated. The hang is in a script
between S01 and network bring-up (S03markos-net), and the box never
transmits a single frame (no gratuitous ARP, no ARP answers — only the
router ever appears in the LAN ARP table).

## Eliminations (each verified on hardware)

| suspect | verdict | evidence |
|---|---|---|
| Boot FAT corruption (installer writer) | real bug, fixed, not the hang | fsck clean after fix; still hung |
| Boot firmware (2026-09-15 Pi OS start4.elf) | real bug, fixed | 2-2 blink died with the old firmware swap |
| AXCL kernel modules | innocent | blacklisted boot still hung |
| Kernel build delta (Image 25.8 MB vs proven 22.9 MB) | real bug, fixed | old Image swap changed behavior (freeze → livelock-blink) |
| fstab / inittab / init.d diffs | benign | full diff of old vs new rootfs |
| Data-partition first-boot grow | innocent (design) | S05 detaches it; network is S03 |

## Remaining suspects (in the new rootfs's early userspace)

1. **S01markos-mounts restructure** — the /etc tmpfs overlay + mount_boot
   rewrite is the largest early-boot delta. v4 swaps the proven-old S01 in.
2. S02markos-recovery (identical to old — unlikely, but traced anyway).
3. Something lower: busybox/s6 binary delta in the rebuilt rootfs.

## v4 (staged on the card, needs one UAC click to flash)

- proven firmware (start4.elf badf4cd8…), proven kernel (Image 5f64d950…)
- **proven-old S01markos-mounts** from the last-known-good image
- **instrumented rcS**: every init script gets `START`/`DONE` markers with
  uptime + rc appended to `/boot/markos/boot-debug.txt` and the console
- axcl modules NOT blacklisted (full stack gets its chance under the old S01)

### Morning procedure

1. Card in PC → run `os\output\flash-v4.bat` (elevate via UAC).
2. Card to Pi, power on, wait 5+ minutes.
3. Either the box answers at http://10.0.0.69:8080/healthz — done — or:
4. Pull the card, plug into PC, read `F:\markos\boot-debug.txt`:
   the last `trace[...]: rcS: START <script>` line that never got its DONE
   is the exact hanging script; lines inside S01 pinpoint further.

## Where the v4 artifacts live

- `os/output/markos-sd-v4.img` — the image (also /root/markos-sd-v4.img in WSL)
- `os/output/flash-v4.bat` — one-click flasher
- WSL `/root/rootfs-v4.squashfs` — the instrumented rootfs blob
- Engine set for the NPU tier: `os/output/axcl-sets/qwen3-0.6b/` (29 .axmodel
  + set.txt from AXERA-TECH/Qwen3-0.6B) — push to `/data/axcl/sets/` once SSH works

## RESOLVED (2026-09-17) — root causes and landed fixes

Every issue on this page was root-caused and fixed in the repo. The
remaining open item is NPU-generation quality (garbage output), not boot.

### Root causes found (in order of discovery)

| # | symptom | root cause | fix landed |
|---|---|---|---|
| 1 | 2-long-2-short LED code (bootloader refuses boot FAT) | installer's FAT32 writer produced spec-invalid volumes (stale FSInfo free count, 0-byte files carrying cluster chains, LFN slots with 0xFFFF cluster fields, bad `..` entries) | `installer/src/fat.rs`: writer fixed + regression test (`f15b4c7`) |
| 2 | Same LED code on the *fixed* FAT | Boot firmware itself: the 2026-09-15 Pi OS `start4.elf` wedges 2026-production d0-stepping boards in the bootloader (2-long-2-short "failed to read partition") | firmware vendored at `os/board/markos/bootfw/` (`bc8394c`) — the post-image script no longer downloads |
| 3 | Kernel boots, S03 hangs without timeouts | `dropbearkey -t rsa` blocks on `getrandom()` — the squashfs root can't persist the seedrng state (`/var/lib/seedrng` on RO root), CRNG starts cold on headless hardware with no entropy sources | `S01seedrng` now seeds from `/data/state/seedrng` (persistent); S03 reordered network-first so keygen can't block the interface; keys persist after first boot |
| 4 | S03 completes but network still dark | **busybox has no `timeout` applet** — every `timeout 5 ip link set eth0 up` silently SKIPPED the command. The static config was decided but never applied to the interface | S03 rewritten with zero timeout wrappers; busybox `CONFIG_TIMEOUT=y` added via `board/markos/busybox-axcl.fragment` |
| 5 | "Daemon already running on PID NNN" console spam | avahi-daemon s6 respawn loop: first instance dies without pid-file cleanup (no usable interface), PID is recycled by another process, libdaemon's `kill(pid,0)` probe says alive → infinite respawn | `avahi/run` script clears the stale pid file before exec |
| 6 | MAC TX queue timeout (netconsole evidence) | Kernel 6.18.52 (rpi-6.18.y pin) has BCM2712 ethernet TX-queue wedging on this board — the first TX wedges the queue, NETDEV WATCHDOG fires at 10s, MAC never recovers | **Kernel re-pinned to 576cc10e (rpi-6.6.y, the hardware-proven kernel)** in defconfig; every prior flash ran 6.6.28-v8-16k without this issue |
| 7 | AXCL driver modules can't load ("Invalid parameters") | Driver built against 6.18.52 but running kernel is 6.6.28 → MODVERSIONS CRC mismatch on kernel symbols (`kmalloc_caches`, `kernel_write` etc.) | Modules rebuilt against the intact 6.6.28 tree (the ssd build dir preserved the matching `Module.symvers`); driver package in Buildroot now targets the re-pinned kernel |
| 8 | net.conf not adopted from boot FAT | Same as #4: the `timeout 5 cp` wrapper skipped the copy command | Same fix (timeout wrappers removed); plus S03's self-heal block re-adopts from the FAT if state is missing |
| 9 | S40network fights S03 ("RTNETLINK: File exists") | `BR2_SYSTEM_DHCP="eth0"` generates an `/etc/network/interfaces` with a dhcp stanza; `ifup -a` runs after S03 and tries to reconfigure the interface | defconfig's `BR2_SYSTEM_DHCP` removed; overlay ships a loopback-only `/etc/network/interfaces` |
| 10 | Data partition doesn't grow to fill card | `resize2fs` fails with "No space left on device while checking for online resizing support" — the 6.6 kernel's ext4 online-resize ioctl has a limitation with the 512 MB → 953 GB jump | Workaround documented: boot once, then manually `mkfs.ext4` the partition at full size and restore state (the appliance's S01 corrupt-data path does exactly this) |

### What's still open

- **NPU generation quality**: with the card present and the engine armed,
  generation produces `????????` instead of text. The engine templates
  load onto the card (CMM populated to 943 MiB) but NPU utilization stays
  at 0% — layers staged but not executing. The CPU-tier path (any GGUF)
  is fully proven on hardware. This needs investigation of the whole-layer
  dispatch/weight-patch path in the fork's backend — likely the Q8_0
  quant requires the `layout_v4.bin` sidecar that the vendor's pre-built
  templates don't include.
