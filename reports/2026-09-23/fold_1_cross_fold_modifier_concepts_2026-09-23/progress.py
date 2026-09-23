"""Keep a readable live report until the final concept report exists."""

import time
from common import HERE, DATE, read, now


def update():
    state = read(HERE / "status.json") if (HERE / "status.json").exists() else {"phase": "starting"}
    phase = state["phase"]
    lines = [f"# Cross-fold modifier concepts — live status, {DATE}", "",
             f"Updated: {now()}", "", f"Phase: **{phase}**", "",
             "Baseline committed on main as `568ef7a` before this experiment.", "",
             "1. Extend each saved fold ranking from 64 to 100 using cached numerical evidence and LLM comparisons.",
             "2. Review their union in one global LLM concept pass, then select existing representative measurements.",
             "3. Freeze selections, evaluate exact oracle identity recovery, and write the final report. No effect estimator is fitted.", "",
             "| Inner fold | State | New comparisons completed | New comparisons started |",
             "| --- | --- | ---: | ---: |"]
    for number in range(1, 6):
        path = HERE / "fold_rankings" / f"fold_{number:03d}_status.json"
        fold = read(path) if path.exists() else {}
        lines.append(f"| {number} | {fold.get('phase', 'waiting')} | {fold.get('completed_requests', 0)} | {fold.get('new_requests', 0)} |")
    lines += ["", "The started comparison count grows as ranking merges progress. It is not the total expected workload; cached comparisons and repair attempts are separate.", ""]
    if phase == "global_concept_review":
        lines += [f"Reviewing **{state['union_candidates']} distinct candidates** across all five top-100 lists.", ""]
    report = HERE / f"REPORT_{DATE}.md"
    if phase == "complete" and report.exists() and (HERE / "experiment_complete.json").exists():
        lines += [f"Completed: {state['modifiers']} representative modifiers from {state['retained_concepts']} retained concepts.",
                  f"Direct oracle modifier recovery: {state['direct_oracle_modifier_recovery']}/5.", "",
                  f"[Final report]({report.name})", ""]
    elif phase == "failed":
        lines += [f"Failure: {state.get('error', 'Inspect the experiment logs.')}", ""]
    else:
        lines += ["The final report will be written only after concept selection and its audit finish.", ""]
    path = HERE / f"STATUS_{DATE}.md"
    temp = path.with_suffix(".tmp")
    temp.write_text("\n".join(lines))
    temp.replace(path)
    return phase


if __name__ == "__main__":
    while update() not in {"complete", "failed"}:
        time.sleep(30)
