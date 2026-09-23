# GPT-6 Sol access diagnosis — 2026-09-23

The installed Codex CLI version caused the observed access failure. GPT-6 Sol works on this account through CLI 0.156.1. The earlier statement that the account lacked model access was too strong: the older client returned a misleading account-support error.

| Probe | Client | Local model catalog | Result |
|---|---|---|---|
| Initial attempt | 0.154.0 | Existing catalog | HTTP 400: model not supported with a ChatGPT account |
| Updated-client test | 0.156.1 | Fresh | Succeeded; GPT-6 Sol returned `READY` |
| Paired old-client check after the success | 0.154.0 | Fresh | Same HTTP 400 failure |

The paired checks use the same account, requested model, minimal prompt, network access, and isolated filesystem. The old-client failure persisted after the newer client succeeded, ruling out a stale host cache alone and supporting a version-dependent catalog/compatibility problem. These tests do not identify the exact internal server implementation.

The [official September 23 release notes](https://learn.chatgpt.com/docs/changelog) identify CLI 0.156.1 as the release adding GPT-6 Sol and Luna to the model catalog. The September 22 announcement describes a staged rollout. Workspace controls can affect access generally, but the successful request establishes access for this account.

For this experiment, an official 0.156.1 package was installed into a temporary directory and used directly. The existing global CLI remains 0.154.0. Updating the installed CLI to 0.156.1 or later is the demonstrated fix for CLI use; the desktop app and an already-running task may also need an update or refresh to expose their current model picker. Desktop picker behavior has not been separately tested.

Evidence: `capability_probe_cli_0_154_0/`, `capability_complete.json`, `capability_events.jsonl`, `old_client_fresh_complete.json`, and `old_client_fresh_events.jsonl`. The actual evidence review uses GPT-6 Sol. No GPT-5.6 review was run.
