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
