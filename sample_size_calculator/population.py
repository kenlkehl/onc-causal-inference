"""Explicit, seeded population parameters. Draw once; edits become the saved truth."""
import math
import random

ROLES = {'confounder': 'C', 'modifier': 'M', 'both': 'B', 'noise': 'N'}
TYPES = ('continuous', 'binary', 'categorical')

def group_specs(cfg):
    rows=[]
    for role,prefix in ROLES.items():
        i=0
        for kind in TYPES:
            for _ in range(cfg['variables'][role][kind]):
                i+=1; rows.append(dict(group=f'{prefix}{i}',role=role,kind=kind))
    return rows

def draw_population(cfg):
    rng=random.Random(cfg['population_seed']); rows=[]
    for spec in group_specs(cfg):
        row=dict(spec,alpha=0.,gamma=0.,beta=0.,prevalence=None,probabilities=None)
        if spec['role'] in ('confounder','both'):
            row['alpha']=rng.gauss(cfg['treatment_coefficient'],cfg['treatment_sd'])
            row['gamma']=rng.gauss(cfg['outcome_coefficient'],cfg['outcome_sd'])
        if spec['role'] in ('modifier','both'):
            row['beta']=rng.gauss(cfg['modifier_strength'],cfg['modifier_sd'])
        if spec['kind']=='binary':
            row['prevalence']=(rng.betavariate(cfg['binary_alpha'],cfg['binary_beta'])
                               if cfg['binary_mode']=='beta' else cfg['binary_prevalence'])
        if spec['kind']=='categorical':
            if cfg['category_mode']=='dirichlet':
                values=[rng.gammavariate(cfg['category_concentration']*p,1.) for p in cfg['category_probs']]
                total=sum(values)
                if total==0: raise ValueError('Dirichlet draw underflow. Increase concentration.')
                row['probabilities']=[v/total for v in values]
            else: row['probabilities']=list(cfg['category_probs'])
        rows.append(row)
    validate_population(cfg,rows)
    return rows

def finite(x): return type(x) in (int,float) and math.isfinite(x)

def validate_population(cfg,rows):
    specs=group_specs(cfg)
    if not isinstance(rows,list) or len(rows)!=len(specs): raise ValueError('Population rows do not match the variable counts. Draw the population again.')
    for row,spec in zip(rows,specs):
        if not isinstance(row,dict) or set(row)!={'group','role','kind','alpha','gamma','beta','prevalence','probabilities'}:
            raise ValueError('Invalid population row.')
        if any(row[k]!=v for k,v in spec.items()): raise ValueError('Population roles/types do not match. Draw the population again.')
        for key in ('alpha','gamma','beta'):
            if not finite(row[key]) or abs(row[key])>10: raise ValueError(f'{row["group"]}: {key} must be finite and within [-10,10]. No draw was clipped; revise the distribution or edit the draw.')
        if spec['role'] not in ('confounder','both') and (row['alpha']!=0 or row['gamma']!=0): raise ValueError('Only confounder/both-role variables may have confounding coefficients.')
        if spec['role'] not in ('modifier','both') and row['beta']!=0: raise ValueError('Only modifier/both-role variables may have an effect coefficient.')
        if spec['kind']=='binary':
            if not finite(row['prevalence']) or not 0<row['prevalence']<1: raise ValueError('Binary prevalence must be strictly between 0 and 1.')
        elif row['prevalence'] is not None: raise ValueError('Only binary variables have prevalence.')
        if spec['kind']=='categorical':
            p=row['probabilities']
            if not isinstance(p,list) or len(p)!=len(cfg['category_probs']) or any(not finite(v) or v<=0 for v in p) or abs(sum(p)-1)>1e-8:
                raise ValueError('Each category probability must be positive and probabilities must sum to 1.')
        elif row['probabilities'] is not None: raise ValueError('Only categorical variables have category probabilities.')
