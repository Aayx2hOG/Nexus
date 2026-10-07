# Nexus frontend

The Next.js console calls `/api/*`; its server-side proxy forwards requests to
FastAPI and keeps the API token out of the browser.

From the repository root, start the model, backend, and frontend together:

```sh
.venv/bin/python scripts/dev.py
```

Open http://localhost:3000/traffic, load a preset or upload a flow CSV, and run
detection. View results in `/alerts` and the loaded release in `/model`.

For separate frontend startup, set `NEXUS_BACKEND_URL` (default
`http://127.0.0.1:8000`) and `NEXUS_API_TOKEN` to match the backend, then run
`npm run dev` from this directory. See [the local guide](../docs/LOCAL_DEMO.md)
for dependency installation and manual backend configuration.
