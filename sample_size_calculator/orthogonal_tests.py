"""Stage 2's orthogonal modifier tests as the calculator's third selector.

Unlike the rest of the calculator, this module needs the OCI environment
(``oci`` with numpy, pandas, and scipy). ``app.py`` starts one worker per run
with ``--serve RUN``; it records its settings and the OCI sources it loaded in
``RUN/orthogonal_tests_worker.json`` and prints its loopback port. For each
simulated study, ``engine.R`` writes the study's labels, nuisances, and encoded
covariates to a temporary directory and sends one request line; the worker
runs ``candidate_tests`` and ``combine_and_select`` with Stage 2's default
settings and writes ``modifier_tests.json`` in the study directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import socket
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from oci.inference import stage2_modifier_tests as tests
from oci.inference.stage2_elastic_net_selection import Stage2ElasticNetSelectionConfig
from oci.inference.stage2_multi_model_config import ModifierSelectionConfig

# PlainHandoffStage2Config's defaults: propensity_clip, and min_propensity and
# max_propensity unset, so every row is eligible. Importing that module takes
# seconds, so tests/test_sample_size_calculator.py checks these copies instead.
PROPENSITY_CLIP = 0.02
LABELS = ('A', 'Y', 'e', 'mu0', 'mu1')


def settings(q):
    return {
        'q': float(q),
        'weights': tests.default_weights('binary'),
        'max_modifiers': ModifierSelectionConfig().max_modifiers,
        'propensity_clip': PROPENSITY_CLIP,
        'eligible_rows': 'all',
        'categorical_min_count': Stage2ElasticNetSelectionConfig().categorical_min_count,
        'tests_schema_version': tests.SCHEMA_VERSION,
    }


def study_inputs(directory, n):
    """Labels, nuisances, and one Stage 2 candidate per original variable.

    A continuous variable is its value; a binary or categorical variable is its
    level (``"1"`` is the reference, as in the simulation), rebuilt from its
    indicator columns.
    """
    with (Path(directory) / 'columns.csv').open(newline='') as handle:
        columns = list(csv.DictReader(handle))
    values = np.fromfile(Path(directory) / 'values.bin', dtype='<f8')
    values = values.reshape(len(LABELS) + len(columns), int(n))
    labels = dict(zip(LABELS, values[:len(LABELS)]))
    parts = {}
    for row, column in zip(columns, values[len(LABELS):]):
        parts.setdefault(row['group'], (row['kind'], []))[1].append((int(row['contrast']), column))
    frame, definitions = {}, []
    for group, (kind, indicators) in parts.items():
        if kind == 'continuous':
            frame[group] = indicators[0][1]
            definitions.append({'feature_id': group, 'name': group, 'value_type': 'continuous'})
        else:
            level = 1 + sum(contrast * column for contrast, column in indicators)
            frame[group] = np.rint(level).astype(int).astype(str)
            definitions.append({
                'feature_id': group, 'name': group, 'value_type': 'categorical',
                'categories_or_unit': [str(k) for k in range(1, len(indicators) + 2)],
            })
    return pd.DataFrame(frame), definitions, labels


def run_study(case, inputs, n, chosen):
    frame, definitions, v = study_inputs(inputs, n)
    records, audit = tests.candidate_tests(
        frame, definitions, treatment=v['A'], outcome=v['Y'], propensity=v['e'],
        mu0=v['mu0'], mu1=v['mu1'], binary=True, propensity_clip=chosen['propensity_clip'],
        min_count=chosen['categorical_min_count'],
    )
    records, selected = tests.combine_and_select(
        records, weights=chosen['weights'], q=chosen['q'], binary=True,
        max_modifiers=chosen['max_modifiers'],
    )
    result = {
        'selected': selected,
        'records': [
            {'feature_id': r['feature_id'], 'selected': r['selected'],
             'grouped_columns': r['grouped_columns'],
             'dropped_sparse_columns': r['dropped_sparse_columns'],
             **tests.compact_evidence(r, rule='orthogonal_tests')}
            for r in records
        ],
        'audit': {key: audit[key] for key in ('rows', 'eligible_rows', 'evaluable')},
    }
    path = Path(case) / 'modifier_tests.json'
    path.write_text(json.dumps(tests.json_ready(result), indent=2, allow_nan=False) + '\n')
    return selected


def provenance(chosen):
    names = ('stage2_modifier_tests', 'stage2_statistical_selection', 'stage2_interaction_screen',
             'stage2_elastic_net_selection', 'stage2_multi_model_config')
    files = [Path(tests.__file__).with_name(f'{name}.py') for name in names]
    return {
        'settings': chosen,
        'python': platform.python_version(),
        'oci_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }


def serve(run):
    chosen = settings(json.loads((Path(run) / 'config.json').read_text())['modifier_q'])
    (Path(run) / 'orthogonal_tests_worker.json').write_text(
        json.dumps(provenance(chosen), indent=2) + '\n')
    # app.py holds stdin open; when it exits, stop rather than outlive it.
    threading.Thread(target=lambda: (sys.stdin.read(), os._exit(0)), daemon=True).start()
    with socket.create_server(('127.0.0.1', 0)) as server:
        print(server.getsockname()[1], flush=True)
        while True:
            connection, _ = server.accept()
            with connection, connection.makefile('rw', encoding='utf-8', newline='\n') as stream:
                case, inputs, n = stream.readline().rstrip('\n').split('\t')
                try:
                    run_study(case, inputs, n, chosen)
                    reply = 'ok'
                except Exception as exc:  # reported to R, which records a failed study
                    reply = 'error ' + ' '.join(f'{type(exc).__name__}: {exc}'.split())
                stream.write(reply + '\n')
                stream.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--serve', type=Path, required=True, metavar='RUN')
    serve(parser.parse_args().serve)


if __name__ == '__main__':
    main()
