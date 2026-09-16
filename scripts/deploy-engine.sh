#!/bin/bash
# Deploy a freshly cross-built engine to the Pi over SSH and restart it.
# The squashfs root is read-only, so the new binary lives on /data and the
# service run script (writable /etc tmpfs overlay) points at it for this
# boot session. Permanent changes go through a normal reflash.
set -e
SSH_OPTS="-o ConnectTimeout=4 -o BatchMode=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
PI=root@10.0.0.69
BIN="${1:-C:/Users/Mark/Desktop/Projects/MarkOS/os/output/markos-engine.pi}"

scp $SSH_OPTS "$BIN" "$PI:/data/markos-engine.new" >/dev/null
ssh $SSH_OPTS "$PI" '
	chmod 755 /data/markos-engine.new
	mv /data/markos-engine.new /data/bin/markos-engine 2>/dev/null || {
		mkdir -p /data/bin && mv /data/markos-engine.new /data/bin/markos-engine
	}
	grep -q /data/bin /etc/service/markos-engine/run || sed -i "s|exec /usr/bin/markos-engine|exec /data/bin/markos-engine|" /etc/service/markos-engine/run
	kill $(pidof markos-engine) 2>/dev/null || true
	sleep 2
	pidof markos-engine && echo DEPLOY-OK || echo DEPLOY-FAILED
'
