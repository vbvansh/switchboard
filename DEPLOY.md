# Putting Switchboard on a public URL

Switchboard serves its own landing page at `/`, so one deploy gives you the
website, the dashboard, the API and the metrics endpoint at a single address.
There is no separate frontend to host.

```
https://your-app.onrender.com/            the landing page
https://your-app.onrender.com/dashboard   spend and savings
https://your-app.onrender.com/health      status
https://your-app.onrender.com/metrics     Prometheus counters
https://your-app.onrender.com/v1/...      the OpenAI-compatible API
```

---

## Before you deploy: three things to know

**1. A hosting platform cannot run your local models.** Ollama needs a machine
with your models on it. A deployed Switchboard has nothing to route to unless
you connect a hosted provider and give it a key. The landing page detects this
and says so rather than letting a visitor find out from a 503.

**2. On a free plan, the ledger is temporary.** Free tiers have no persistent
disk, so the SQLite file is wiped on every deploy and every restart. Users,
budgets and spending history go with it. Fine for a public demo; attach
PostgreSQL for anything real. Both configurations are already written out in
[render.yaml](render.yaml) — one is commented out.

**3. The dashboard is not behind a password.** That is deliberate — it shows
aggregate spend and model names, never prompt text or keys — but on a public
URL, anyone with the link can see it. If that is not acceptable, put the
service behind your platform's access control, or do not deploy the dashboard
publicly.

---

## Render (recommended)

### One-time setup

```powershell
# Make sure everything is pushed first
git add .
git commit -m "Add deployment configuration"
git push
```

Then, in a browser:

1. Sign in at [render.com](https://render.com) with your GitHub account.
2. **New → Blueprint**.
3. Pick the `switchboard` repository. Render finds `render.yaml` on its own.
4. **Apply**. The first build takes about five minutes — it is installing
   scikit-learn and friends into the image.
5. When it finishes, Render shows a URL like
   `https://switchboard-xxxx.onrender.com`. That is your public site.

### Connecting a model provider

Without one, the API has nowhere to send requests. In the Render dashboard:

**Environment → Add Environment Variable**

| Key | Value |
|---|---|
| `OPENROUTER_API_KEY` | your key from [openrouter.ai](https://openrouter.ai) |

Then edit `providers.yaml` in the repository: set `enabled: true` on the
`openrouter` provider, and add some models under it. To find them:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
python -m switchboard discover openrouter --contains "claude" --limit 5
```

That prints a YAML block with real prices. Paste it under the provider's
`models:` key, commit, and push — Render redeploys automatically.

### Creating a user on the deployed instance

API keys live in the database, so they have to be created on the machine that
has the database. Render gives you a shell:

**Your service → Shell**

```bash
python -m switchboard users add alice --budget 25
```

Copy the key it prints. It is shown once and never again — only its hash is
stored.

### Free plan behaviour

The service sleeps after 15 minutes of no traffic, and the next request takes
about 30 seconds while it wakes. That is normal, and it is why the health check
in `render.yaml` points at `/health/live`.

---

## Railway

Nearly identical, and it reads the same `Dockerfile`.

1. [railway.app](https://railway.app) → **New Project → Deploy from GitHub repo**.
2. Railway builds the Dockerfile automatically.
3. **Settings → Networking → Generate Domain** for a public URL.
4. **Variables**: add `SWITCHBOARD_HOST` = `0.0.0.0`.
5. For a database that survives deploys: **New → Database → PostgreSQL**, then
   set `SWITCHBOARD_DATABASE_URL` to
   `postgresql+psycopg://...` using the credentials Railway shows you.

Railway sets `PORT` itself; the entrypoint already reads it.

---

## Fly.io

```powershell
fly launch --no-deploy      # detects the Dockerfile, writes fly.toml
fly volumes create switchboard_data --size 1
```

In the generated `fly.toml`:

```toml
[env]
  SWITCHBOARD_HOST = "0.0.0.0"
  SWITCHBOARD_DATABASE_URL = "sqlite:////app/data/switchboard.db"

[[mounts]]
  source = "switchboard_data"
  destination = "/app/data"

[http_service]
  internal_port = 8000

[[http_service.checks]]
  path = "/health/live"
```

Then `fly deploy`. The volume is what keeps the ledger across deploys.

---

## Your own server

Most control, and the only option where nothing is hidden from you. On any
Linux box with Docker installed and a domain pointed at it:

```bash
git clone https://github.com/vbvansh/switchboard
cd switchboard
docker compose up -d
```

That serves plain HTTP on port 8000. For HTTPS, put Caddy in front — it obtains
and renews certificates on its own:

```
# /etc/caddy/Caddyfile
switchboard.yourdomain.com {
    reverse_proxy localhost:8000
}
```

```bash
sudo systemctl reload caddy
```

---

## Checking it worked

```powershell
# Is the process alive?
Invoke-RestMethod https://your-app.onrender.com/health/live

# What does it think its state is?
Invoke-RestMethod https://your-app.onrender.com/health | ConvertTo-Json -Depth 5
```

`"status": "degraded"` with an empty `providers` map means the service is
running and has no model connected — expected until you add a provider key.

---

## If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Deploy times out, "no open ports" | The server bound to `127.0.0.1` | Set `SWITCHBOARD_HOST=0.0.0.0` |
| Service restarts in a loop | Health check pointed at `/health/ready`, which is 503 with no provider | Point it at `/health/live` |
| Startup fails: "schema out of date" | Migrations did not run | Check `SWITCHBOARD_AUTO_MIGRATE` is not `false` |
| Users disappear after a deploy | SQLite on an ephemeral disk | Attach PostgreSQL, or a persistent volume |
| `/v1/chat/completions` returns 503 | No provider configured | Add a provider key and enable it in `providers.yaml` |
| Requests return 401 | No API key, or a wrong one | `switchboard users add <name>` in the platform shell |

---

## What is deliberately not here

**No sign-up or user accounts on the website.** The site is informational; API
keys are created by an administrator with `switchboard users add`. Adding web
sign-up means passwords, sessions, email verification and account recovery —
a real security surface that deserves its own phase rather than being tacked
onto a landing page.
