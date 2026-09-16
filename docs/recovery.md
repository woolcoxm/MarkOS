# MarkOS — recovery paths

A headless box with no monitor is only acceptable if there is always a way
back in. MarkOS ships three independent, implemented recovery paths; any one
of them works even when the web-UI config or LAN networking is broken.

## 1. Link-local rescue UI (no tools needed)

The engine always binds the web UI additionally on
`http://169.254.9.1:4444`, and the OS assigns that IPv4LL address to the
first Ethernet interface at every boot — unconditionally, before any DHCP or
static config.

**Procedure:** plug a laptop straight into the Pi's Ethernet port (no
switch needed), wait ~10 s, open `http://169.254.9.1:4444`. Fix networking
under *Network & exposure*, or use *System → Restart engine*. The nftables
firewall explicitly allows the `169.254.0.0/16` subnet.

## 2. Factory-reset button (GPIO 26)

Hold a button wired between **GPIO 26 and GND** from power-on. The boot
stage (`S02markos-recovery`) polls the pin; 3 seconds of hold triggers:

- revert any pending A/B update (`/boot/tryboot-cmdline.txt` removed),
- wipe `/data/state` (engine config, users, network config) — **models on
  `/data/models` are preserved**,
- restore the provision snapshot from `/boot/markos-snapshot/` (the
  install-time config, copied there on first boot),
- reboot into a box that behaves exactly like a fresh install.

A reset can also be triggered without the button from the web UI
(*System → Recovery* writes `/data/state/recovery-requested`, honored at
next boot) or on-device with `markos-factory-reset`.

## 3. Serial console (UART)

The debug UART is enabled in `config.txt` (`dtparam=uart0=on`) and a getty
runs on `ttyAMA10` at **115200 8N1** — J8 header pins 8 (TXD) / 10 (RXD) /
6 (GND). From any USB-serial adapter you get the console: boot logs,
a root shell (the root account is password-locked; use the *runsv* services
through the console to disable dropbear's flag file or edit
`/data/state/net.conf` by hand).

## What recovery covers

| failure | path back |
|---|---|
| Bad static IP / DHCP broken | 1 (link-local) or 3 (edit net.conf) |
| Web-UI config breaks the engine | supervision restarts it; config fix via 1 |
| Forgotten admin password | 2 (factory reset restores install-time admin) |
| Failed/aborted A/B update | automatic (tryboot revert) or 2 |
| Root filesystem corruption | SD: squashfs is read-only, cannot corrupt; SSD: A/B slot revert |
| Supervisor itself hangs | hardware watchdog reboots the box (`watchdogd`, 60 s timeout) |

## Deliberate non-features

No cloud fallback, no "phone home for help" — every recovery path is local
and offline. See `docs/design.md` §6 for the rationale.
