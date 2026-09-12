# Stage 2 sampling defaults

Stage 2 looks up a checked-in publisher profile after resolving the serving model
through `/models`. It uses the advertised backing model when available, so a
served alias can still select the appropriate model/version. Primary and
extraction endpoints resolve separately. Profiles were verified on 2026-09-12;
inference does not need internet access or scrape mutable model cards.

| Model | Mode | Temperature | Top-p | Top-k | Presence penalty | Repetition penalty |
| --- | --- | --- | --- | --- | --- | --- |
| Gemma 4 | Either | 1.0 | 0.95 | 64 | 0 | 1.0 |
| Qwen 3 | Thinking | 0.6 | 0.95 | 20 | 0 | 1.0 |
| Qwen 3 | Non-thinking | 0.7 | 0.8 | 20 | 0 | 1.0 |
| Qwen 3.5 | Thinking | 1.0 | 0.95 | 20 | 1.5 | 1.0 |
| Qwen 3.6 / 3.8 | Thinking | 1.0 | 0.95 | 20 | 0 | 1.0 |
| Qwen 3.5 / 3.6 / 3.8 | Non-thinking | 0.7 | 0.8 | 20 | 1.5 | 1.0 |
| LFM 2.5 2.6B | Either | 0.1 | 1.0 | 50 | 0 | 1.1 |
| LFM 2.5 1.2B | Either | 0.1 | 1.0 | 50 | 0 | 1.05 |

Sources: [Google Gemma 4](https://huggingface.co/google/gemma-4-31B-it#best-practices),
[Qwen 3](https://huggingface.co/Qwen/Qwen3-32B#best-practices),
[Qwen 3.5](https://huggingface.co/Qwen/Qwen3.5-27B#best-practices),
[Qwen 3.6](https://huggingface.co/Qwen/Qwen3.6-27B#best-practices),
[Qwen 3.8](https://huggingface.co/Qwen/Qwen3.8-27B#best-practices),
[Liquid 2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B), and
[Liquid 1.2B](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct).

Recognized profiles explicitly disable unspecified penalties and filters:
frequency penalty 0, min-p 0, and the neutral values shown above. These neutral
values are pipeline choices where the publisher gives no recommendation, not
claims that every publisher explicitly documents every field. In particular,
Google recommends temperature/top-p/top-k but does not prescribe a nonzero
presence penalty. Repetition penalty 1.0 means no repetition penalty.
Unrecognized families omit sampling fields and defer to the server.

Omitted or null `stage2.temperature`, `top_p`, `top_k`, `min_p`,
`presence_penalty`, `frequency_penalty`, and `repetition_penalty` select automatic
defaults. Explicit numeric values override the profile for both primary and
extraction requests, including an explicit temperature of zero. For example:

```json
{"stage2": {"temperature": null, "presence_penalty": 0.5, "repetition_penalty": null}}
```

Thinking-dependent defaults are recalculated if a repair turns thinking on.
Resolved sampling values appear in request logs and the request policies used
for scientific checkpoint fingerprints. Changing sampling settings can invalidate
completed Stage 2 checkpoints; old explicitly configured values remain overrides
until removed or set to null. Stage 1 artifacts are unaffected.

Compatible endpoints receive top-k, min-p, and repetition penalty through
`extra_body`. An endpoint rejecting these extensions can reach the existing
logged compatibility fallback, which omits them and uses its server defaults.
