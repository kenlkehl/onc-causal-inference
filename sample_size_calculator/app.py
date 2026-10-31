"""Local, dependency-free web/CLI front end to the calculator's R selection methods."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from .population import draw_population, validate_population

HERE = Path(__file__).resolve().parent
METHODS = HERE / 'binary_selection_methods.R'
RUNS = HERE.parent / 'artifacts' / 'sample_size_calculator'
DOCS = 'docs/sample_size_calculator.md'
R_INSTALL = 'Rscript -e \'install.packages(c("glmnet", "jsonlite"), repos="https://cloud.r-project.org")\''
DEFAULTS = dict(
    variables=dict(confounder=dict(continuous=5,binary=0,categorical=0),
                   modifier=dict(continuous=3,binary=1,categorical=1),
                   both=dict(continuous=0,binary=0,categorical=0),
                   noise=dict(continuous=100,binary=0,categorical=0)),
    scale='logOR',modifier_strength=.6,treatment_coefficient=.5,outcome_coefficient=.5,
    baseline_risk=.3,treated_fraction=.5,constant_effect=.2,binary_prevalence=.5,
    category_probs=[.5,.3,.2],correlation=0.,goal='all',recovery_fraction=.8,
    false_allowance=1,target_probability=.8,nuisance='estimated',
    sample_sizes=[400,800],replications=8,seed=910100,calibration_seed=910000,mode='pilot',
    schema_version=2,sensitivity=.8,false_positive_rate=.05,
    modifier_sd=0.,treatment_sd=0.,outcome_sd=0.,population_seed=920000,
    binary_mode='beta',binary_alpha=5.,binary_beta=5.,category_mode='dirichlet',category_concentration=20.,
    population=None,selector='lasso',modifier_q=.1)

def validate(raw):
    if not isinstance(raw,dict) or set(raw)-set(DEFAULTS):
        raise ValueError('Unknown configuration fields.')
    cfg=copy.deepcopy(DEFAULTS); cfg.update(raw)
    cfg['schema_version']=raw.get('schema_version',1)
    if cfg['schema_version'] not in (1,2): raise ValueError('Unsupported configuration version.')
    def number(key,low,high,integer=False):
        x=cfg[key]
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not low<=x<=high or (integer and int(x)!=x):
            raise ValueError(f'{key} must be {"an integer " if integer else ""}between {low} and {high}.')
    for key,choices in dict(scale=['logOR','RD'],goal=['all','fraction'],nuisance=['estimated','oracle'],mode=['pilot','grid'],selector=['lasso','orthogonal_tests']).items():
        if cfg[key] not in choices: raise ValueError(f'Invalid {key}.')
    for key,lo,hi in [('modifier_strength',-5,5),('treatment_coefficient',-5,5),('outcome_coefficient',-5,5),
                      ('baseline_risk',.05,.95),('treated_fraction',.1,.9),('constant_effect',-2,2),
                      ('binary_prevalence',.05,.95),('correlation',0,.8),('recovery_fraction',.01,1),
                      ('target_probability',.5,.99),('modifier_q',.001,1)]: number(key,lo,hi)
    for key,lo,hi in [('false_allowance',0,300),('replications',1,2500),('seed',1,10000000),('calibration_seed',1,10000000)]: number(key,lo,hi,True)
    if cfg['mode']=='pilot' and cfg['replications']>30: raise ValueError('A pilot allows at most 30 replications per size.')
    if cfg['mode']=='grid' and cfg['replications']<200: raise ValueError('Grid mode requires at least 200 replications per size.')
    sizes=cfg['sample_sizes']
    if not isinstance(sizes,list) or not 1<=len(sizes)<=8 or any(type(n)!=int or not 200<=n<=20000 for n in sizes):
        raise ValueError('Enter 1–8 distinct integer sample sizes between 200 and 20,000.')
    if len(set(sizes))!=len(sizes): raise ValueError('Sample sizes must be distinct.')
    cfg['sample_sizes']=sorted(sizes)
    # engine.R's study_seed reserves 1000 seeds per study and 2500 studies per size.
    if cfg['seed']<=cfg['calibration_seed']<cfg['seed']+2500000*len(sizes):
        raise ValueError(f'calibration_seed must lie outside the simulation seeds, seed to seed + {2500000*len(sizes)-1:,}.')
    probs=cfg['category_probs']
    if not isinstance(probs,list) or not 2<=len(probs)<=5 or any(type(v) not in (int,float) or not math.isfinite(v) or v<.05 for v in probs) or abs(sum(probs)-1)>1e-8:
        raise ValueError('Category probabilities: 2–5 numbers, each at least 0.05, summing to 1.')
    variables=cfg['variables']; coords=0; groups=0
    if not isinstance(variables,dict) or set(variables)!=set(DEFAULTS['variables']): raise ValueError('Invalid role table.')
    for row in variables.values():
        if not isinstance(row,dict) or set(row)!={'continuous','binary','categorical'}: raise ValueError('Invalid variable types.')
        for kind,count in row.items():
            if type(count)!=int or not 0<=count<=200: raise ValueError('Variable counts must be integers from 0 to 200.')
            groups+=count; coords+=count*(len(probs)-1 if kind=='categorical' else 1)
    if not 2<=groups<=220 or coords>300: raise ValueError('Use 2–220 candidate groups and at most 300 encoded columns.')
    if min(sizes)<=coords+30: raise ValueError('Smallest n must exceed encoded columns by at least 31 for the reused full-rank Zhao calibration.')
    if cfg['scale']=='RD' and abs(cfg['constant_effect'])+cfg['modifier_strength']>1:
        raise ValueError('RD effects are probability differences, not log odds. Try strength 0.03 and constant RD 0.02.')
    if cfg['schema_version']==1 and cfg['population'] is not None: raise ValueError('Legacy configurations cannot include v2 population rows.')
    if cfg['schema_version']==2:
        for key,lo,hi in [('sensitivity',0,1),('false_positive_rate',0,1),
                          ('modifier_sd',0,2),('treatment_sd',0,2),('outcome_sd',0,2),
                          ('binary_alpha',.1,1000),('binary_beta',.1,1000),('category_concentration',.5,1000)]: number(key,lo,hi)
        number('population_seed',1,10000000,True)
        if cfg['binary_mode'] not in ('fixed','beta') or cfg['category_mode'] not in ('fixed','dirichlet'): raise ValueError('Invalid prevalence distribution.')
        if cfg['correlation']!=0: raise ValueError('Version 2 uses independent covariates; correlation must be zero.')
        if cfg['population'] is not None: validate_population(cfg,cfg['population'])
    return cfg

def save_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

OCI_PROBE=None

def check_oci():
    """The orthogonal_tests selector runs Stage 2's code in this interpreter."""
    global OCI_PROBE
    if OCI_PROBE is None:
        done=subprocess.run([sys.executable,'-c','import oci.inference.stage2_modifier_tests'],capture_output=True,text=True)
        OCI_PROBE=done.stderr.strip().splitlines()[-1] if done.returncode else ''
    if OCI_PROBE:
        raise ValueError(f'The orthogonal_tests selector needs the OCI environment ({OCI_PROBE}). '
                         'Run uv sync --frozen, then start the calculator with PYTHON=.venv/bin/python ./run_sample_size_calculator.sh.')

def create_run(cfg,path):
    cfg=validate(cfg)
    if cfg['selector']=='orthogonal_tests': check_oci()
    if cfg['schema_version']==2 and cfg['population'] is None: cfg['population']=draw_population(cfg)
    path=Path(path).resolve(); path.mkdir(parents=True,exist_ok=False)
    save_json(path/'config.json',cfg)
    (path/'sources').mkdir()
    for source in [HERE/'app.py',HERE/'engine.R',HERE/'run.R',HERE/'index.html',HERE/'app.js',HERE/'population.py',HERE/'orthogonal_tests.py',METHODS]:
        shutil.copy2(source,path/'sources'/source.name)
    manifest=dict(started=datetime.now(timezone.utc).isoformat(),configuration=cfg,
        python=platform.python_version(),source_hashes={p.name:sha(p) for p in (path/'sources').iterdir()},
        protocol=(DOCS if cfg['schema_version']==2 else DOCS+'#legacy-configurations'),outputs={},state='running')
    save_json(path/'manifest.json',manifest)
    save_json(path/'progress.json',dict(state='running',done=0,total=len(cfg['sample_sizes'])*cfg['replications']))
    save_json(path/'control.json',dict(action='running'))
    return path

def start_worker(path,env):
    """One orthogonal-tests worker per run; R sends it each study over loopback."""
    with (path/'orthogonal_tests.log').open('w') as log:
        worker=subprocess.Popen([sys.executable,str(path/'sources/orthogonal_tests.py'),'--serve',str(path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env,text=True)
    port=worker.stdout.readline().strip()
    if not port.isdigit():
        worker.kill(); worker.communicate()
        raise RuntimeError('The orthogonal-tests worker did not start; see orthogonal_tests.log.')
    return worker,port

def execute(path):
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']: env[key]='1'
    worker=None
    try:
        if json.loads((path/'config.json').read_text())['selector']=='orthogonal_tests':
            worker,env['OCI_CALCULATOR_TESTS_PORT']=start_worker(path,env)
        with (path/'execution.log').open('w') as log:
            done=subprocess.run(['Rscript',str(path/'sources/run.R'),str(path/'config.json'),str(path),str(path/'sources/binary_selection_methods.R')],stdout=log,stderr=subprocess.STDOUT,env=env)
        if done.returncode:
            progress=json.loads((path/'progress.json').read_text())
            if progress.get('state')!='failed': save_json(path/'progress.json',dict(state='failed',error='R execution failed; see execution.log.'))
    except Exception as exc:
        save_json(path/'progress.json',dict(state='failed',error=str(exc)))
    finally:
        if worker is not None: worker.kill(); worker.communicate()
    manifest=json.loads((path/'manifest.json').read_text())
    manifest.update(finished=datetime.now(timezone.utc).isoformat(),state=json.loads((path/'progress.json').read_text())['state'],
                    outputs={str(p.relative_to(path)):sha(p) for p in path.rglob('*') if p.is_file() and p.name!='manifest.json'})
    save_json(path/'manifest.json',manifest)

LOCK=threading.Lock(); ACTIVE=None

def run_path(name,removed=False):
    if not isinstance(name,str) or not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):
        raise ValueError('Invalid run id.')
    root=RUNS/'.removed' if removed else RUNS
    path=root/name
    if root.is_symlink() or path.is_symlink(): raise ValueError('Linked run directories are not supported.')
    return path

def list_runs(removed=False):
    root=RUNS/'.removed' if removed else RUNS
    items=[]
    if root.exists() and not root.is_symlink():
        for path in sorted(root.iterdir(),reverse=True):
            if path.is_dir() and not path.is_symlink() and (path/'progress.json').exists():
                record=json.loads((path/'progress.json').read_text())
                manifest=json.loads((path/'manifest.json').read_text()) if (path/'manifest.json').exists() else {}
                items.append(dict(record,id=path.name,removable=record.get('state') in ('complete','failed','cancelled') and manifest.get('state') in ('complete','failed','cancelled')))
    return items

def move_runs(ids,restore=False):
    """Move frozen runs without changing any contents or hashes; preflight the batch."""
    if not isinstance(ids,list) or not ids or len(ids)>500 or any(not isinstance(x,str) for x in ids) or len(set(ids))!=len(ids):
        raise ValueError('Select 1–500 distinct runs.')
    with LOCK:
        moves=[]
        for name in ids:
            source=run_path(name,removed=restore); destination=run_path(name,removed=not restore)
            if not source.is_dir() or not (source/'progress.json').is_file(): raise ValueError(f'Run not found: {name}')
            progress=json.loads((source/'progress.json').read_text())
            manifest=json.loads((source/'manifest.json').read_text()) if (source/'manifest.json').exists() else {}
            if progress.get('state') not in ('complete','failed','cancelled') or manifest.get('state') not in ('complete','failed','cancelled'):
                raise ValueError(f'{name}: running or unfinished runs cannot be moved.')
            if destination.exists(): raise ValueError(f'Destination already exists: {name}')
            moves.append((source,destination))
        completed=[]
        try:
            for source,destination in moves:
                destination.parent.mkdir(parents=True,exist_ok=True)
                source.rename(destination); completed.append((source,destination))
        except OSError:
            for source,destination in reversed(completed): destination.rename(source)
            raise
    return dict(ids=ids,action='restored' if restore else 'removed')

def control_run(payload):
    if not isinstance(payload,dict) or set(payload)!={'id','action'} or payload['action'] not in ('running','paused','cancelled'):
        raise ValueError('Invalid run control.')
    with LOCK:
        path=run_path(payload['id'])
        if not (path/'control.json').exists(): raise ValueError('This run does not support live controls.')
        progress=json.loads((path/'progress.json').read_text())
        if progress.get('state') not in ('running','paused'): raise ValueError('This run has already stopped.')
        previous=json.loads((path/'control.json').read_text())
        if previous.get('action')=='cancelled': raise ValueError('Cancellation is already requested.')
        action=payload['action']
        save_json(path/'control.json',dict(action=action,requested=datetime.now(timezone.utc).isoformat()))
    return dict(id=path.name,action=action)

class Handler(BaseHTTPRequestHandler):
    def send(self,value,status=200,content_type='application/json'):
        body=(json.dumps(value,allow_nan=False).encode() if content_type=='application/json' else value)
        self.send_response(status); self.send_header('Content-Type',content_type); self.send_header('Content-Length',str(len(body)))
        self.send_header('Cache-Control','no-store'); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path=='/': return self.send((HERE/'index.html').read_bytes(),content_type='text/html; charset=utf-8')
        if self.path=='/app.js': return self.send((HERE/'app.js').read_bytes(),content_type='text/javascript; charset=utf-8')
        if self.path=='/api/defaults': return self.send(DEFAULTS)
        if self.path in ('/api/runs','/api/runs/removed'):
            return self.send(list_runs(removed=self.path.endswith('/removed')))
        if self.path.startswith('/api/run/'):
            name=self.path.removeprefix('/api/run/')
            try: path=run_path(name)
            except ValueError as exc: return self.send({'error':str(exc)},400)
            if not (path/'progress.json').exists(): return self.send({'error':'Not found'},404)
            value=json.loads((path/'progress.json').read_text()); value['id']=name; value['path']=str(path)
            if (path/'control.json').exists(): value['control']=json.loads((path/'control.json').read_text())
            value['config']=json.loads((path/'config.json').read_text()) if (path/'config.json').exists() else None
            if (path/'result.json').exists(): value['result']=json.loads((path/'result.json').read_text())
            return self.send(value)
        return self.send({'error':'Not found'},404)
    def do_POST(self):
        global ACTIVE
        if self.path not in ('/api/run','/api/runs/remove','/api/runs/restore','/api/population','/api/control'): return self.send({'error':'Not found'},404)
        # Loopback-only service, JSON-only requests and same-origin writes.
        if self.headers.get('Content-Type','').split(';')[0]!='application/json': return self.send({'error':'JSON required'},415)
        origin=self.headers.get('Origin')
        if origin and origin!=f'http://{self.headers.get("Host")}': return self.send({'error':'Origin mismatch'},403)
        try:
            length=int(self.headers.get('Content-Length',0))
            if not 0<length<250000: raise ValueError('Invalid request size')
            payload=json.loads(self.rfile.read(length))
            if self.path=='/api/population':
                cfg=validate(dict(payload,population=None))
                return self.send(dict(population=draw_population(cfg)))
            if self.path=='/api/control': return self.send(control_run(payload))
            if self.path!='/api/run':
                if not isinstance(payload,dict) or set(payload)!={'ids'}: raise ValueError('Expected selected run ids.')
                return self.send(move_runs(payload['ids'],restore=self.path.endswith('/restore')))
            cfg=validate(payload)
            with LOCK:
                if ACTIVE is not None and ACTIVE.is_alive(): return self.send({'error':'One simulation is already running. Follow its progress or wait for completion.'},409)
                name=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid.uuid4().hex[:6]
                path=create_run(cfg,RUNS/name)
                ACTIVE=threading.Thread(target=execute,args=(path,),daemon=True); ACTIVE.start()
            return self.send(dict(id=name),202)
        except (ValueError,KeyError,TypeError) as exc: return self.send({'error':str(exc)},400)
        except OSError as exc: return self.send({'error':str(exc)},409)
    def log_message(self,*args): pass

def check_r():
    """Fail before any run starts when Rscript or its packages are missing."""
    if shutil.which('Rscript') is None: raise SystemExit(f'Rscript is not on PATH. Install R, then: {R_INSTALL}  (see {DOCS})')
    probe="quit(status=!all(vapply(c('glmnet','jsonlite'),requireNamespace,logical(1),quietly=TRUE)))"
    if subprocess.run(['Rscript','-e',probe],capture_output=True).returncode:
        raise SystemExit(f'The R packages glmnet and jsonlite are required: {R_INSTALL}  (see {DOCS})')

def main():
    global RUNS
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--port',type=int,default=8765)
    p.add_argument('--config',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--runs-dir',type=Path,default=RUNS,help=f'Dashboard run directory (default: {RUNS})')
    args=p.parse_args()
    RUNS=args.runs_dir.resolve()
    check_r()
    if args.config:
        if not args.output: p.error('--output is required with --config')
        path=create_run(json.loads(args.config.read_text()),args.output); execute(path)
        state=json.loads((path/'progress.json').read_text()); print(json.dumps(state)); print(path)
        raise SystemExit(0 if state['state']=='complete' else 1)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Calculator: http://127.0.0.1:{args.port}  (runs saved under {RUNS})',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
if __name__=='__main__': main()
