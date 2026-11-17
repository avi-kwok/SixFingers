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
It is safe to run again any time. It installs packages, builds the Python venv, downloads the three backbone weights, installs the `sixfingers-api` service, Caddy, the firewall rules, the nightly cron entry, and the `sixfingers` command. It takes a while on the first run.

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
The API needs `models/current`. Either copy a trained model dir from your Mac, or train on the VM:
```bash
rsync -a models/v1 ubuntu@<vm-ip>:~/SixFingers/models/   # from your Mac
ssh ubuntu@<vm-ip> 'cd ~/SixFingers/models && ln -sfn v1 current && sudo systemctl restart sixfingers-api'
```

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

## Everyday operations
| Task | Command |
|---|---|
| API logs | `journalctl -u sixfingers-api -f` |
| Caddy logs | `journalctl -u caddy -f` |
| Restart API | `sudo systemctl restart sixfingers-api` |
| Nightly loop log | `tail -n 50 ~/SixFingers/data/loop.log` |
| Run loop by hand | `cd ~/SixFingers && bash deploy/run_loop.sh` |
| Roll back the model | `cd ~/SixFingers && .venv/bin/python -m ml.loop.run rollback` (or `rollback v3`) |
| Update code | `cd ~/SixFingers && git pull && bash deploy/setup.sh` |

The API notices the `models/current` symlink changing and reloads by itself, so a rollback needs no restart.

## Troubleshooting
- `tls cert` FAIL: DNS record missing or not propagated, or ports 80/443 closed (check step 2, then `sudo iptables -L INPUT -n --line-numbers`).
- `api model loaded` FAIL: `models/current` missing, or read `journalctl -u sixfingers-api -n 50`.
- Loop never reports: check `/etc/cron.d/sixfingers-loop` exists and `data/loop.log`.
