"""Compare old CLI with the same fresh-catalog isolation that enabled the new CLI."""

import datetime as dt
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
manifest = json.loads((HERE / "input_manifest.json").read_text())
launch = json.loads((HERE / "capability_launch.json").read_text())
old_binary = str(Path.home() / ".local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex")
command = [old_binary if arg == manifest["codex_binary"] else arg for arg in launch["command"]]
command = ["/work/capability_old_fresh_response.txt" if arg == "/work/capability_response.txt" else arg for arg in command]
env = {key: value for key, value in os.environ.items() if key in {
    "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR",
}}
env["PATH"] = "/usr/bin:/bin"
record = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
          "cli_version": "0.154.0", "model": "gpt-6-sol", "catalog": "fresh, no host cache mounted", "command": command}
with (HERE / "old_client_fresh_events.jsonl").open("wb") as events, (HERE / "old_client_fresh_stderr.log").open("wb") as errors:
    result = subprocess.run(command, env=env, input=b"Return exactly the word READY. Do not use tools.\n", stdout=events, stderr=errors, timeout=120)
record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), returncode=result.returncode)
(HERE / "old_client_fresh_complete.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps({key: value for key, value in record.items() if key != "command"}, indent=2))
