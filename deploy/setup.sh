#!/usr/bin/env bash
# Idempotent setup for Ubuntu ARM (Oracle A1). Run as the "ubuntu" user from the repo: bash deploy/setup.sh
# Safe to re-run: every step checks before changing anything. Never touches secrets.
set -euo pipefail

DOMAIN="${SIXFINGERS_DOMAIN:-api.sixfingers.avikwok.dev}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$(id -un)}"
[ "$RUN_USER" = root ] && { echo "run as the normal user (not root); sudo is used where needed" >&2; exit 1; }
say() { printf '\n== %s\n' "$*"; }

say "apt packages"
sudo apt-get update -y
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common
# Ubuntu 22.04/24.04 do not ship python3.11 in the default repos; deadsnakes does.
apt-cache show python3.11-venv >/dev/null 2>&1 || { sudo add-apt-repository -y ppa:deadsnakes/ppa && sudo apt-get update -y; }
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3.11 python3.11-venv python3-pip git curl util-linux debian-keyring debian-archive-keyring \
  apt-transport-https gpg iptables iptables-persistent netfilter-persistent libgl1 libglib2.0-0

say "python venv + dependencies (CPU torch on the server)"
cd "$REPO"
[ -x .venv/bin/python ] || python3.11 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .

say "backbone weights (cached under ~/.cache; skipped if present)"
.venv/bin/python - <<'PY'
import open_clip
from huggingface_hub import snapshot_download
open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
snapshot_download("facebook/dinov2-large")
snapshot_download("google/siglip-base-patch16-224")
print("weights ok")
PY

say "directories + .env placeholder (permissions only; secrets are a manual step, see RUNBOOK)"
mkdir -p data models
[ -f .env ] || { umask 177; : > .env; }
chmod 600 .env

say "systemd unit sixfingers-api"
sudo tee /etc/systemd/system/sixfingers-api.service >/dev/null <<UNIT
[Unit]
Description=SixFingers API
After=network-online.target
Wants=network-online.target

[Service]
User=$RUN_USER
WorkingDirectory=$REPO
EnvironmentFile=$REPO/.env
ExecStart=$REPO/.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
Restart=on-failure
RestartSec=5
Nice=5

[Install]
WantedBy=multi-user.target
UNIT
sudo systemctl daemon-reload
sudo systemctl enable sixfingers-api >/dev/null
sudo systemctl restart sixfingers-api

say "Caddy (auto-HTTPS for $DOMAIN)"
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y caddy
fi
sed "s/__DOMAIN__/$DOMAIN/" "$REPO/deploy/Caddyfile" | sudo tee /etc/caddy/Caddyfile >/dev/null
sudo systemctl enable caddy >/dev/null
sudo systemctl reload caddy || sudo systemctl restart caddy

say "firewall: open 80/443 (Oracle Ubuntu images reject them by default) and persist"
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport "$p" -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 5 -p tcp --dport "$p" -j ACCEPT
done
sudo netfilter-persistent save

say "nightly loop cron (02:00 Pacific; flock + timeout + nice; ends before 06:00)"
sudo tee /etc/cron.d/sixfingers-loop >/dev/null <<CRON
CRON_TZ=America/Los_Angeles
0 2 * * * $RUN_USER flock -n /tmp/sixfingers-loop.lock timeout 230m nice -n 19 bash $REPO/deploy/run_loop.sh >> $REPO/data/loop.log 2>&1
CRON
sudo chmod 644 /etc/cron.d/sixfingers-loop

say "sixfingers doctor command"
sudo tee /usr/local/bin/sixfingers >/dev/null <<CLI
#!/usr/bin/env bash
cd "$REPO" && exec .venv/bin/python deploy/doctor.py "\$@"
CLI
sudo chmod 755 /usr/local/bin/sixfingers

say "done. Next: sixfingers doctor"
