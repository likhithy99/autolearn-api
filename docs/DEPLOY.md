# Putting the API on the internet

Customers can only buy and use keys once the API is reachable over **HTTPS** at a stable URL.
Two free routes follow, then the paid upgrade path. Free-tier offers change, so check the current terms.

## Option A: free cloud VM (recommended for launch)

**Oracle Cloud "Always Free"** offers an Arm (Ampere A1) VM with up to 4 cores and 24 GB RAM at no
cost. That's enough for the API, AutoML and small-LLM inference.

```bash
# on the VM (Ubuntu 22.04+, Arm)
sudo apt update && sudo apt install -y python3.12-venv git caddy
git clone <your repo> autolearn-api && cd autolearn-api
python3.12 -m venv .venv
.venv/bin/pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# secrets live in an env file readable only by you
cat > .env <<'EOF'
AUTOLEARN_ADMIN_KEY=<long random string>
AUTOLEARN_DATA_DIR=/home/ubuntu/autolearn-data
AUTOLEARN_LLM_DIR=/home/ubuntu/autolearn-models
AUTOLEARN_PUBLIC_URL=https://api.yourdomain.com
STRIPE_SECRET_KEY=sk_live_...
STRIPE_WEBHOOK_SECRET=whsec_...
STRIPE_PRICE_PRO=price_...
EOF
chmod 600 .env
```

`/etc/systemd/system/autolearn.service`:

```ini
[Unit]
Description=AutoLearn API
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/autolearn-api
EnvironmentFile=/home/ubuntu/autolearn-api/.env
ExecStart=/home/ubuntu/autolearn-api/.venv/bin/uvicorn autolearn.main:app --host 127.0.0.1 --port 8000 --workers 1 --proxy-headers
Restart=always

[Install]
WantedBy=multi-user.target
```

`/etc/caddy/Caddyfile` (Caddy fetches and renews the TLS certificate automatically):

```
api.yourdomain.com {
    reverse_proxy 127.0.0.1:8000
}
```

```bash
sudo systemctl enable --now autolearn caddy
```

Point the domain's DNS `A` record at the VM's public IP. Open ports 80 and 443 in both the Oracle
security list and the VM firewall. A domain costs about $10/year; it's the only cost.

## Option B: run from your Mac with a free Cloudflare Tunnel (beta / demos)

```bash
brew install cloudflared
cloudflared tunnel --url http://localhost:8000      # prints a public https://*.trycloudflare.com URL
```

This takes zero setup and gives free HTTPS. The quick-tunnel URL changes on every restart; a named
tunnel on your own domain (free Cloudflare account) gives a stable URL. The API is down whenever the
Mac sleeps or is offline, so use this for demos and early pilot customers, not paying production traffic.

## Production checklist

* [ ] HTTPS only (Caddy or Cloudflare). Never serve plain HTTP to customers.
* [ ] `AUTOLEARN_ADMIN_KEY` is long and random, and stored only in `.env` or a secret manager.
* [ ] Stripe live keys set; webhook endpoint `https://api.yourdomain.com/stripe/webhook` registered.
* [ ] Daily backup of `AUTOLEARN_DATA_DIR` and the LLM registry (for example `rclone` to free object
      storage such as Cloudflare R2's free tier).
* [ ] Uptime check on `/health` (free: UptimeRobot, Better Stack).
* [ ] Prometheus scraping `/metrics` (with the platform key), or at least log monitoring.
* [ ] Terms of service, privacy policy, and acceptable-use policy linked from the pricing page.
* [ ] Set `AUTOLEARN_SALES_EMAIL` to your real contact address (shown on the pricing page).

## Scaling up later (paid)

| Stage | Setup |
|---|---|
| First paying customers | One VM (Option A) |
| Growing traffic | Several API containers behind a load balancer; move SQLite to Postgres and model files to object storage |
| Large LLM | GPU instance (`Dockerfile.gpu`) or a dedicated inference server (vLLM) behind this API |
