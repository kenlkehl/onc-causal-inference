"""Run one fresh Sol reviewer with only blinded inputs mounted into its filesystem."""

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "input_manifest.json").read_text())
MODEL = MANIFEST.get("review_model", MANIFEST["requested_model"])
WORK = Path(MANIFEST["isolation_directory"]) / "work"
HOME_PATH = Path.home()
CODEX_BINARY = Path(MANIFEST.get("codex_binary", str(HOME_PATH / ".local/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex")))


def sandbox_command():
    command = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts"]
    for path in ("/usr", "/lib", "/lib64", "/etc"):
        if Path(path).exists():
            command += ["--ro-bind", path, path]
    resolver = Path("/etc/resolv.conf").resolve()
    if not str(resolver).startswith("/etc/"):
        command += ["--ro-bind", str(resolver), str(resolver)]
    command += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"]
    command += ["--dir", str(HOME_PATH / ".codex")]
    # The CLI performs its normal account authentication. Never inspect credentials.
    # A fresh model catalog rules out stale local model metadata.
    for filename in ("auth.json",):
        source = HOME_PATH / ".codex" / filename
        if source.exists():
            command += ["--ro-bind", str(source), str(source)]
    command += ["--ro-bind", str(CODEX_BINARY), "/opt/codex"]
    command += ["--bind", str(WORK), "/work", "--chdir", "/work"]
    return command


def main():
    mode = sys.argv[1]
    if mode not in {"probe", "capability", "review"}:
        raise SystemExit("mode must be probe, capability, or review")
    env = {key: value for key, value in os.environ.items() if key in {
        "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "TZ",
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR",
    }}
    env["PATH"] = "/usr/bin:/bin"
    command = sandbox_command()
    if mode == "probe":
        probe = "import json,pathlib; print(json.dumps({'repository_mounted':pathlib.Path('/data1').exists(),'home_entries':[p.name for p in pathlib.Path.home().iterdir()],'work_entries':[p.name for p in pathlib.Path('/work').iterdir()]}))"
        result = subprocess.run(command + ["/usr/bin/python3", "-c", probe], env=env, capture_output=True, text=True, check=True)
        (HERE / "isolation_probe.json").write_text(result.stdout)
        print(result.stdout)
        return
    command += ["/opt/codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                "--sandbox", "read-only", "--cd", "/work", "--json", "--color", "never",
                "--model", MODEL, "-c", 'model_reasoning_effort="high"',
                "-c", "model_context_window=1050000", "-c", "model_auto_compact_token_limit=950000",
                "-c", 'web_search="disabled"', "-c", "mcp_servers={}"]
    for feature in ("shell_tool", "apps", "plugins", "hooks", "memories", "multi_agent", "view_image",
                    "image_generation", "browser_use", "computer_use", "workspace_dependencies", "skill_search"):
        command += ["--disable", feature]
    command += ["--enable", "skip_host_skill_discovery"]
    if mode == "review":
        command += ["--output-schema", "/work/schema.json"]
    command += ["--output-last-message", f"/work/{mode}_response.txt", "-"]
    record = {"started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "mode": mode,
              "model": MODEL, "reasoning_effort": "high", "command": command,
              "input_sha256": hashlib.sha256((WORK / ("prompt.txt" if mode == "review" else "capability_prompt.txt")).read_bytes()).hexdigest()}
    (HERE / f"{mode}_launch.json").write_text(json.dumps(record, indent=2) + "\n")
    with (WORK / ("prompt.txt" if mode == "review" else "capability_prompt.txt")).open("rb") as prompt, (HERE / f"{mode}_events.jsonl").open("wb") as events, (HERE / f"{mode}_stderr.log").open("wb") as errors:
        result = subprocess.run(command, env=env, stdin=prompt, stdout=events, stderr=errors, timeout=3600)
    record.update(ended_utc=dt.datetime.now(dt.timezone.utc).isoformat(), returncode=result.returncode)
    response = WORK / f"{mode}_response.txt"
    if response.exists():
        shutil.copyfile(response, HERE / response.name)
        record["response_sha256"] = hashlib.sha256(response.read_bytes()).hexdigest()
    (HERE / f"{mode}_complete.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "command"}, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
