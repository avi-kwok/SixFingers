# SixFingers deploy runbook

Target: Oracle Cloud Ubuntu VM (ARM A1.Flex), API at `https://api.sixfingers.avikwok.dev`, front end on Vercel at `https://sixfingers.avikwok.dev`.

Steps marked **(manual step)** need cloud accounts or secrets and are done by hand. Everything else is scripted.

## 1. Resize the VM (manual step)
1. Oracle Cloud console, Compute, Instances, your instance, Edit shape.
2. Set OCPUs to 3 and memory to 18 GB (still inside the always-free allowance). The instance reboots.
3. Check the boot volume is large enough (50 GB or more). Model weights and data need roughly 15 GB.

## 2. Open the cloud firewall (manual step)
Oracle blocks traffic twice: once in the cloud (security list) and once inside the VM (iptables; `setup.sh` handles that one).
1. Networking, Virtual cloud networks, your VCN, Security Lists, Default.
2. Add two ingress rules: source `0.0.0.0/0`, TCP, destination port `80`, then the same for `443`.

## 3. DNS (manual step)
Add an A record: `api.sixfingers.avikwok.dev` pointing to the VM's public IP. For the web app, add the record Vercel tells you in step 8 (usually a CNAME for `sixfingers`). Wait until `dig +short api.sixfingers.avikwok.dev` shows the IP. Caddy cannot get a certificate before this works.

## 4. Get the code onto the VM
```bash
ssh ubuntu@<vm-ip>
git clone https://github.com/avi-kwok/SixFingers.git ~/SixFingers
```
(The repo must be pushed first.)

## 5. Run setup
```bash
cd ~/SixFingers && bash deploy/setup.sh
```
It is safe to run again any time. It installs packages, builds the Python venv, downloads the three backbone weights, installs the `sixfingers-api` service, Caddy, the firewall rules and the `sixfingers` command. It takes a while on the first run.

It also writes `SIXFINGERS_TRUST_PROXY=1` to `.env`. That tells the API to read the visitor's IP from the `X-Forwarded-For` header Caddy adds, so each visitor gets their own rate limit. (uvicorn already trusts that header from 127.0.0.1; the setting makes the app's own rule explicit.) It is only safe because Caddy is the only way in: the API listens on `127.0.0.1:8000`, which the internet cannot reach.

The nightly self-training loop is **off** by default. On a small VM it would train on a tiny data folder, could replace a good model with a weaker one, and could run out of memory. See [Nightly loop (optional)](#nightly-loop-optional) to turn it on.

## 6. Enter secrets (manual step)
The prompts below do not echo what you type, and nothing lands in shell history.
```bash
cd ~/SixFingers
read -rs -p "Anthropic API key: " K; echo; printf 'ANTHROPIC_API_KEY=%s\n' "$K" >> .env; unset K
read -rs -p "Discord webhook URL: " W; echo; printf 'DISCORD_WEBHOOK_URL=%s\n' "$W" >> .env; unset W
chmod 600 .env
sudo systemctl restart sixfingers-api
```
Run each block once. If you mistype, edit `.env` with `nano .env` (do not `cat` it on screen). Set a console spend limit on the Anthropic key's workspace as a second safety net.

## 7. Load a model
The API serves whatever `models/current` points to. Copy the model your Mac uses now (run from `~/SixFingers` on your Mac):
```bash
V=$(readlink models/current)   # e.g. v15
rsync -a "models/$V" ubuntu@<vm-ip>:~/SixFingers/models/
ssh ubuntu@<vm-ip> "cd ~/SixFingers/models && ln -sfn $V current && sudo systemctl restart sixfingers-api"
```
`api/judge_calibration.json` is in git, so it arrives with the clone. If you changed it and have not pushed, copy it too: `rsync -a api/judge_calibration.json ubuntu@<vm-ip>:~/SixFingers/api/`.

`sixfingers doctor` prints the version the API is serving next to `models/current` and warns if they differ.

## 8. Vercel project (manual step)
1. vercel.com, Add New Project, import the GitHub repo.
2. Root directory: `web`.
3. Environment variable: `NEXT_PUBLIC_API_URL` = `https://api.sixfingers.avikwok.dev`.
4. Deploy, then Settings, Domains, add `sixfingers.avikwok.dev` and create the DNS record Vercel shows.

## 9. Check everything
```bash
sixfingers doctor
```
Each line is OK, WARN or FAIL. Fix FAILs first. `curl -s https://api.sixfingers.avikwok.dev/v1/health` should answer.

## Settings in `.env`
The API reads these at start. After a change, run `sudo systemctl restart sixfingers-api`.

| Name | Default | What it does |
|---|---|---|
| `ANTHROPIC_API_KEY` | none | Key for the Claude Haiku 5.5 judge. Without it, answers come from the detector only. |
| `DISCORD_WEBHOOK_URL` | none | Where the nightly loop posts its report. |
| `SIXFINGERS_TRUST_PROXY` | `1` (set by setup) | Take the visitor IP from Caddy's `X-Forwarded-For` header. |
| `SIXFINGERS_JUDGE_BUDGET` | `500` | Judge calls allowed per UTC day, for the whole site. |
| `SIXFINGERS_JUDGE_PER_IP_DAY` | `20` | Judge calls allowed per visitor IP per day. After that the visitor gets the detector-only answer. |
| `SIXFINGERS_MAX_CONCURRENT` | `2` | Checks the API runs at the same time. Keeps the small VM from running out of memory. |

## Nightly loop (optional)
Off by default. To turn it on, run setup with the flag:
```bash
cd ~/SixFingers && SIXFINGERS_ENABLE_LOOP=1 bash deploy/setup.sh
```
It runs at 09:00 UTC (01:00 Pacific in winter, 02:00 in summer). Cron on Ubuntu ignores `CRON_TZ`, so the time is written in UTC. Running `bash deploy/setup.sh` without the flag turns the loop off again, so keep the flag on every re-run while you want it.

## Everyday operations
| Task | Command |
|---|---|
| API logs | `journalctl -u sixfingers-api -f` |
| Caddy logs | `journalctl -u caddy -f` |
| Restart API | `sudo systemctl restart sixfingers-api` |
| Nightly loop log | `tail -n 50 ~/SixFingers/data/loop.log` |
| Run loop by hand | `cd ~/SixFingers && bash deploy/run_loop.sh` |
| Roll back the model | `cd ~/SixFingers && .venv/bin/python -m ml.loop.run rollback` (or `rollback v3`) |
| Update code | `cd ~/SixFingers && git pull && bash deploy/setup.sh` (add `SIXFINGERS_ENABLE_LOOP=1` before `bash` if the loop is on) |

The API notices the `models/current` symlink changing and reloads by itself, so a rollback needs no restart.

## Troubleshooting
- `tls cert` FAIL: DNS record missing or not propagated, or ports 80/443 closed (check step 2, then `sudo iptables -L INPUT -n --line-numbers`).
- `api model loaded` FAIL: the API has no model (`models/current` missing) or does not answer; read `journalctl -u sixfingers-api -n 50`. WARN "still loading": the API has just started and is loading the backbones; run the doctor again in a minute. WARN "serving": the API serves a different version than `models/current`; restart it.
- `SIXFINGERS_TRUST_PROXY` FAIL: uvicorn already takes the visitor IP from Caddy's `X-Forwarded-For` (it trusts `127.0.0.1`), but the setting makes the API's own client-IP logic explicit. Run `bash deploy/setup.sh` again, then restart the API.
- `timezone` WARN: the server clock is not UTC, so the loop's cron time `0 9 * * *` runs at the wrong hour. Run `sudo timedatectl set-timezone UTC`.
- Loop never reports: it is off unless you turned it on. Check `/etc/cron.d/sixfingers-loop` exists and read `data/loop.log`.
