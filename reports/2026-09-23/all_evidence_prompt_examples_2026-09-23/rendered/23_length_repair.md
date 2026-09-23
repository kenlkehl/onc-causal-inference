# 23. Repair an overlong response

**Unabridged current template, with invented miniature inputs. No LLM was called.**

Activation: Shared transport: completion length exceeded

Preserve required records and fields; shorten redundant content. Example error text is invented.

Source: [_repair_message](/data1/ken/pcori_dev/causal-dragonnet-text/oci/inference/plain_handoff_stage2.py:3274)

## User message

```text
The previous JSON exceeded the available response length. _Stage2OutputLengthError: response reached the configured completion-token limit. Return one materially shorter corrected JSON object using the same required schema. Remove redundancy, merge duplicate entries, and keep descriptions and rationales concise. Do not omit required records or fields. Return JSON only.
```
