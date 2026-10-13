"""Patient-level evidence, repeatable partitions, and conservative measurement groups.

Only supplied training measurements and observed treatment/outcome labels enter
these operations. Uncertainty summaries are diagnostics, not calibrated tests.
"""
import base64
from collections import defaultdict
from hashlib import sha256
import re
import zlib

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold


def pack(values):
    values = np.asarray(values, dtype='<f8').reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError('nonfinite patient loss differences')
    return {'encoding': 'zlib_float64_le', 'count': len(values),
            'data': base64.b64encode(zlib.compress(values.tobytes())).decode('ascii')}


def unpack(value):
    if value.get('encoding') != 'zlib_float64_le':
        raise ValueError('unsupported patient evidence encoding')
    result = np.frombuffer(zlib.decompress(base64.b64decode(value['data'])), dtype='<f8')
    if len(result) != value['count'] or not np.isfinite(result).all():
        raise ValueError('invalid patient evidence length or values')
    return result


def repeated_partitions(splits, labels, *, treatment_column, outcome_column, binary, repeats, seed):
    """Preserve the original partition; generate additional partitions of its pool."""
    result = [{**s, 'inner_fold': int(s.get('inner_fold', i)), 'partition_repeat': 0, 'partition_fold': i}
              for i, s in enumerate(splits, 1)]
    ids = np.asarray(sorted(labels.index), dtype=int)
    nfolds = len(splits)
    last_original_fold = max(s['inner_fold'] for s in result)
    if nfolds < 2:
        raise ValueError('evidence stability requires at least two folds')
    treatment = labels.loc[ids, treatment_column].astype(str)
    strata = treatment + ':' + labels.loc[ids, outcome_column].astype(str) if binary else treatment
    if strata.value_counts().min() < nfolds:
        strata = treatment
    stratified = strata.value_counts().min() >= nfolds
    for repeat in range(1, repeats):
        splitter = (StratifiedKFold if stratified else KFold)(
            n_splits=nfolds, shuffle=True, random_state=int(seed) + 1_000_003 * repeat)
        for fold, (train, valid) in enumerate(splitter.split(ids, strata), 1):
            result.append({'inner_fold': last_original_fold + (repeat - 1) * nfolds + fold, 'partition_repeat': repeat,
                           'partition_fold': fold, 'fit_row_ids': ids[train].tolist(),
                           'heldout_row_ids': ids[valid].tolist()})
    return result


def _tokens(feature, *, leaf=False):
    # Remove only presentation words; anatomical sites, negation, units and
    # time qualifications remain significant. Broad clinical themes are not groups.
    name = str(feature['name']).lower()
    if leaf:
        name = name.split('__')[-1]
    words = re.findall(r'[a-z0-9]+', name)
    return frozenset(w for w in words if w not in {'measurement', 'value', 'level'})


def measurement_groups(train, definitions, config):
    """Complete-link groups require both similar names and agreeing observations."""
    if not config.group_equivalent_measurements:
        return []
    definitions = sorted(definitions, key=lambda f: f['feature_id'])
    tokens = {f['feature_id']: _tokens(f) for f in definitions}
    by_id = {f['feature_id']: f for f in definitions}
    inverted = defaultdict(list)
    for key, words in tokens.items():
        for word in words:
            inverted[word].append(key)
    possible = set()
    for keys in inverted.values():
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                possible.add((left, right))
    values = {}
    edges = set()
    numeric_types = {'continuous', 'ordinal'}
    for left, right in sorted(possible):
        a, b = by_id[left], by_id[right]
        # Generated child names include their parent theme (e.g. ALK/ROS1).
        # Shared parent tokens or correlation cannot equate distinct targets.
        if _tokens(a, leaf=True) != _tokens(b, leaf=True):
            continue
        if a.get('value_type') != b.get('value_type') or a.get('value_type') == 'ambiguous':
            continue
        ta, tb = tokens[left], tokens[right]
        if not ta or len(ta & tb) / len(ta | tb) < config.group_min_name_similarity:
            continue
        if a.get('value_type') in numeric_types and a.get('categories_or_unit', []) != b.get('categories_or_unit', []):
            continue
        for key, feature in ((left, a), (right, b)):
            if key not in values:
                series = train[feature['name']]
                values[key] = pd.to_numeric(series, errors='coerce') if feature.get('value_type') in numeric_types else series
        x, y = values[left], values[right]
        keep = x.notna() & y.notna()
        n = int(keep.sum())
        if n < config.group_min_observations or n / max(int(x.notna().sum()), int(y.notna().sum()), 1) < config.group_min_overlap:
            continue
        if a.get('value_type') in numeric_types:
            # Positive monotone agreement; inverse or unrelated measurements
            # with strong negative association are not interchangeable.
            association = x[keep].corr(y[keep], method='spearman') if x[keep].nunique() > 1 and y[keep].nunique() > 1 else 0.0
        else:
            association = float((x[keep].astype(str).str.lower() == y[keep].astype(str).str.lower()).mean()) if x[keep].astype(str).nunique() > 1 else 0.0
        if np.isfinite(association) and association >= config.group_min_association:
            edges.add((left, right))
    groups = []
    for key in by_id:
        for group in groups:
            if all(tuple(sorted((key, member))) in edges for member in group):
                group.append(key)
                break
        else:
            groups.append([key])
    return [{'group_id': 'measurement_' + sha256('|'.join(g).encode()).hexdigest()[:16],
             'members': g, 'criterion': 'name_similarity_and_training_observation_agreement_complete_link'}
            for g in groups if len(g) > 1]


def balanced_subsets(feature_ids, groups, *, subset_size, seed):
    """Each feature appears once per partition; alternate measures stay together."""
    available = set(feature_ids)
    units = [g['members'] for g in groups if set(g['members']) <= available]
    grouped = {k for unit in units for k in unit}
    units += [[k] for k in sorted(available - grouped)]
    order = np.random.default_rng(seed).permutation(len(units))
    result, current = [], []
    for index in order:
        unit = units[index]
        if current and len(current) + len(unit) > subset_size:
            result.append(current)
            current = []
        current.extend(unit)
    if current:
        result.append(current)
    return result


def permutation_evidence(x, column_ids, loss, *, repeats, seed, groups=()):
    """Paired validation losses; each feature/group uses the same shuffles."""
    baseline_rows = np.asarray(loss(x), float).reshape(-1)
    baseline = float(baseline_rows.mean())
    rng = np.random.default_rng(seed)
    permutations = [rng.permutation(len(x)) for _ in range(repeats)]
    masks = {key: np.flatnonzero(np.asarray(column_ids) == key) for key in dict.fromkeys(column_ids)}
    masks.update({g['group_id']: np.flatnonzero(np.isin(column_ids, g['members'])) for g in groups if set(g['members']) <= set(column_ids)})
    result = {}
    for key, columns in masks.items():
        deltas = []
        for permutation in permutations:
            changed = x.copy()
            changed[:, columns] = x[permutation][:, columns]
            deltas.append(np.asarray(loss(changed), float).reshape(-1) - baseline_rows)
        means = np.asarray(deltas).mean(axis=1)
        result[key] = {'gain': float(means.mean()), 'sd': float(means.std()),
                       'loss_deltas': pack(np.asarray(deltas).mean(axis=0))}
    return baseline, result


def group_record(feature_id, groups, gains):
    group = next((g for g in groups if feature_id in g['members'] and g['group_id'] in gains), None)
    if group is None:
        return {}
    gain = gains[group['group_id']]
    return {'measurement_group': group, 'group_score': gain['gain'],
            'group_loss_deltas': gain['loss_deltas']}


def patient_summary(rows, field='loss_deltas'):
    """Average repeat exposures per patient first, retaining split sensitivity."""
    patients, partitions = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for row in rows:
        if field not in row or row[field] is None:
            continue
        ids = row['validation_row_ids']
        values = unpack(row[field])
        if len(ids) != len(values) or len(set(ids)) != len(ids):
            raise ValueError('patient evidence does not match unique validation rows')
        for patient, value in zip(ids, values):
            patients[patient].append(float(value))
            partitions[row.get('partition_repeat', 0)][patient].append(float(value))
    if not patients:
        return None
    average = lambda x: float(np.mean(sorted(x)))
    values = [average(patients[p]) for p in sorted(patients)]
    split_means = [float(np.mean([average(patients[p]) for p in sorted(patients)])) for _, patients in sorted(partitions.items())]
    se = float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    split_sd = float(np.std(split_means, ddof=1)) if len(split_means) > 1 else 0.0
    return {'mean_gain': float(np.mean(values)), 'patients': len(values),
            'patient_standard_error_heuristic': se, 'partition_mean_gains': split_means,
            'partition_standard_deviation': split_sd, 'noise_scale': max(se, split_sd),
            'independent_replications_assumed': False, 'formal_error_guarantee': False}
