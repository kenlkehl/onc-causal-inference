"""Modifier-selection sample-size calculator (sample_size_calculator/).

Input, population, run-management, and dashboard checks need only Python. The
R engine checks and the end-to-end launcher runs are skipped unless Rscript and
the R packages glmnet and jsonlite are installed, and the orthogonal-tests
selector checks unless this interpreter has the OCI environment. Runs under
pytest or ``python3 -m unittest tests.test_sample_size_calculator``.
"""
import copy
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from sample_size_calculator import app as calculator
from sample_size_calculator.app import (
    DEFAULTS, check_r, control_run, create_run, list_runs, move_runs, sha, validate)
from sample_size_calculator.population import draw_population

ROOT = Path(__file__).resolve().parents[1]
CALCULATOR = ROOT / 'sample_size_calculator'
THREADS = {key: '1' for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')}


def r_available():
    try:
        check_r()
    except SystemExit:
        return False
    return True


R_AVAILABLE = r_available()
try:
    from sample_size_calculator import orthogonal_tests
except ImportError:
    orthogonal_tests = None


class InputContract(unittest.TestCase):
    def test_defaults_and_configs(self):
        validate(DEFAULTS)
        for path in (CALCULATOR/'configs').glob('*.json'):
            validate(json.loads(path.read_text()))
    def test_impossible_requests(self):
        for change in [dict(sample_sizes=[400,400]),dict(sample_sizes=[100]),dict(replications=31),
                       dict(category_probs=[.5,.6]),dict(treatment_coefficient=float('nan')),
                       dict(mode='grid',replications=8),dict(seed=True),dict(extra='unknown'),
                       dict(selector='gao'),dict(modifier_q=0),dict(calibration_seed=DEFAULTS['seed']+2500000*2-1)]:
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate(dict(copy.deepcopy(DEFAULTS),**change))
    def test_preserves_custom_calibration_and_signs(self):
        c=dict(copy.deepcopy(DEFAULTS),calibration_seed=1234,treatment_coefficient=-.5,outcome_coefficient=.8)
        self.assertEqual(validate(c),c)
    def test_manifest_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=create_run(DEFAULTS,Path(tmp)/'run')
            record=json.loads((run/'manifest.json').read_text())
            for name,expected in record['source_hashes'].items():
                self.assertEqual(sha(run/'sources'/name),expected)
            self.assertIn('binary_selection_methods.R',record['source_hashes'])
            self.assertTrue((ROOT/record['protocol']).is_file())
            self.assertEqual(record['configuration']['variables'],DEFAULTS['variables'])
            self.assertEqual(len(record['configuration']['population']),110)
            self.assertIsNone(DEFAULTS['population'])
            with self.assertRaises(FileExistsError): create_run(DEFAULTS,run)
    def test_missing_oci_refuses_orthogonal_tests_before_creating_a_run(self):
        with tempfile.TemporaryDirectory() as tmp,patch('sample_size_calculator.app.OCI_PROBE','ModuleNotFoundError: no oci'):
            with self.assertRaises(ValueError) as raised:
                create_run(dict(DEFAULTS,selector='orthogonal_tests'),Path(tmp)/'run')
            self.assertIn('uv sync',str(raised.exception)); self.assertFalse((Path(tmp)/'run').exists())
            self.assertTrue(create_run(DEFAULTS,Path(tmp)/'run').is_dir())
    def test_missing_r_names_the_install_command(self):
        with patch('sample_size_calculator.app.shutil.which',return_value=None),self.assertRaises(SystemExit) as raised:
            check_r()
        self.assertIn('install.packages',str(raised.exception))

class PopulationContract(unittest.TestCase):
    def config(self):
        c=copy.deepcopy(DEFAULTS)
        c['variables']={r:dict(continuous=0,binary=0,categorical=0) for r in c['variables']}
        c['variables']['both']['binary']=1;c['variables']['modifier']['categorical']=1
        c['modifier_sd']=.2;c['treatment_sd']=.1;c['outcome_sd']=.15
        return c
    def test_seed_edit_and_zero_truth(self):
        c=self.config();a=draw_population(c)
        self.assertEqual(a,draw_population(c))
        a[0]['beta']=0.;c['population']=a
        self.assertEqual(validate(c)['population'],a)
        self.assertAlmostEqual(math.log(math.exp(-.4)),-.4,places=14)
    def test_role_and_probability_rejection(self):
        c=self.config();c['population']=draw_population(c)
        for key,value in [('alpha',1),('probabilities',[.8,.3,-.1]),('beta',float('nan'))]:
            bad=copy.deepcopy(c);bad['population'][0][key]=value
            with self.assertRaises(ValueError):validate(bad)
        bad=copy.deepcopy(c);bad['schema_version']=1
        with self.assertRaises(ValueError):validate(bad)
    def test_no_prevalence_truncation(self):
        c=self.config();c['population']=draw_population(c)
        c['population'][0]['probabilities']=[.001,.009,.99]
        c['population'][1]['prevalence']=.001
        validate(c)
    def test_legacy_keeps_scalar_mechanism(self):
        c=dict(DEFAULTS);c.pop('schema_version');c['population']=None
        self.assertEqual(validate(c)['schema_version'],1)
    def test_distribution_moments(self):
        c=self.config();rows=[]
        for seed in range(1,4001):
            c['population_seed']=seed;rows.append(draw_population(c))
        beta=[r[0]['beta'] for r in rows]
        prev=[r[1]['prevalence'] for r in rows]
        p0=[r[0]['probabilities'][0] for r in rows]
        self.assertAlmostEqual(statistics.mean(beta),c['modifier_strength'],delta=.012)
        self.assertAlmostEqual(statistics.stdev(beta),.2,delta=.012)
        self.assertAlmostEqual(statistics.mean(prev),.5,delta=.012)
        self.assertAlmostEqual(statistics.variance(prev),25/(100*11),delta=.003)
        self.assertAlmostEqual(statistics.mean(p0),.5,delta=.012)
        self.assertAlmostEqual(statistics.variance(p0),.25/21,delta=.002)

class RunManagement(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.patch=patch('sample_size_calculator.app.RUNS',self.root); self.patch.start(); self.addCleanup(self.patch.stop)
    def fixture(self,name,state='complete',manifest_state=None):
        path=self.root/name; path.mkdir()
        (path/'progress.json').write_text(json.dumps(dict(state=state)))
        (path/'manifest.json').write_text(json.dumps(dict(state=manifest_state or state)))
        (path/'control.json').write_text(json.dumps(dict(action='running')))
        (path/'result.json').write_text('{"sentinel": "preserve exactly"}\n')
        return path
    def test_bulk_remove_restore_preserves_all_bytes(self):
        paths=[self.fixture('first'),self.fixture('second','failed')]
        before={p.name:{f.name:f.read_bytes() for f in p.iterdir()} for p in paths}
        move_runs(['first','second'])
        self.assertEqual(list_runs(),[])
        self.assertEqual({r['id'] for r in list_runs(True)},{'first','second'})
        move_runs(['first','second'],restore=True)
        self.assertEqual(list_runs(True),[])
        self.assertEqual(before,{p.name:{f.name:f.read_bytes() for f in p.iterdir()} for p in paths})
    def test_batch_preflight_protects_running_and_finalizing(self):
        self.fixture('good'); self.fixture('running','running'); self.fixture('finalizing',manifest_state='running')
        for protected in ['running','finalizing']:
            with self.assertRaises(ValueError): move_runs(['good',protected])
            self.assertTrue((self.root/'good').is_dir())
        self.assertEqual([r['id'] for r in list_runs() if r['removable']],['good'])
    def test_controls_and_terminal_protection(self):
        path=self.fixture('live','running')
        for action in ['paused','running','cancelled']:
            control_run(dict(id='live',action=action))
            self.assertEqual(json.loads((path/'control.json').read_text())['action'],action)
        with self.assertRaises(ValueError): control_run(dict(id='live',action='running'))
        self.fixture('done')
        with self.assertRaises(ValueError): control_run(dict(id='done',action='paused'))
    def test_invalid_names_symlinks_and_collisions(self):
        self.fixture('good')
        (self.root/'linked').symlink_to(self.root/'good',target_is_directory=True)
        for ids in [['../good'],['good','good'],['linked'],['missing'],[],['good',1]]:
            with self.subTest(ids=ids),self.assertRaises(ValueError): move_runs(ids)
        move_runs(['good']); self.fixture('good')
        with self.assertRaises(ValueError): move_runs(['good'],restore=True)
        self.assertTrue((self.root/'.removed/good/result.json').is_file())

class Dashboard(unittest.TestCase):
    """The page, its script, and the endpoints that need no R."""
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        runs=patch('sample_size_calculator.app.RUNS',Path(self.tmp.name)); runs.start(); self.addCleanup(runs.stop)
        self.server=ThreadingHTTPServer(('127.0.0.1',0),calculator.Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True).start()
        self.addCleanup(self.server.server_close); self.addCleanup(self.server.shutdown)
        self.url=f'http://127.0.0.1:{self.server.server_address[1]}'
    def request(self,path,payload=None):
        data=None if payload is None else json.dumps(payload).encode()
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=10) as response: return response.read()
    def test_page_defaults_population_and_runs(self):
        page=self.request('/')
        self.assertIn(b'<script src="/app.js">',page); self.assertIn(b'<option value="orthogonal_tests">',page)
        self.assertTrue(self.request('/app.js').startswith(b'const $='))
        self.assertEqual(json.loads(self.request('/api/defaults')),json.loads(json.dumps(DEFAULTS)))
        population=json.loads(self.request('/api/population',DEFAULTS))['population']
        self.assertEqual(population,draw_population(validate(DEFAULTS)))
        self.assertEqual(json.loads(self.request('/api/runs')),[])

def write_study(directory,groups,labels):
    """engine.R's request layout: columns.csv and values.bin (labels, then columns)."""
    import numpy as np
    rows=[(f'x{k:03d}',group,kind,j+1) for k,(group,kind,j,_) in enumerate(groups)]
    with (directory/'columns.csv').open('w') as handle:
        handle.write('"coordinate","group","kind","contrast"\n')
        handle.writelines(f'"{c}","{g}","{kind}",{j}\n' for c,g,kind,j in rows)
    np.concatenate([*labels,*(column for *_,column in groups)]).astype('<f8').tofile(directory/'values.bin')

@unittest.skipUnless(orthogonal_tests is not None,'the OCI environment is not installed in this interpreter')
class OrthogonalTests(unittest.TestCase):
    def test_settings_are_stage2_defaults(self):
        import dataclasses
        from oci.inference.plain_handoff_stage2 import PlainHandoffStage2Config
        from oci.inference.stage2_multi_model_config import ModifierSelectionConfig
        stage2={f.name:f.default for f in dataclasses.fields(PlainHandoffStage2Config)}
        self.assertEqual(orthogonal_tests.PROPENSITY_CLIP,stage2['propensity_clip'])
        self.assertIsNone(stage2['min_propensity']); self.assertIsNone(stage2['max_propensity'])
        self.assertEqual(DEFAULTS['modifier_q'],ModifierSelectionConfig().q)
    def test_levels_and_selection_of_one_study(self):
        import numpy as np
        rng=np.random.default_rng(7); n=3000
        m=rng.uniform(-3**.5,3**.5,n); level=rng.choice(3,n,p=[.5,.3,.2]); b=rng.binomial(1,.4,n)
        noise=[rng.uniform(-3**.5,3**.5,n) for _ in range(4)]
        e=np.full(n,.5); mu0=1/(1+np.exp(-(-.5+.5*noise[0])))
        mu1=1/(1+np.exp(-(np.log(mu0/(1-mu0))+.2+1.2*m)))
        a=rng.binomial(1,e).astype(float); y=rng.binomial(1,np.where(a==1,mu1,mu0)).astype(float)
        groups=[('M1','continuous',0,m),('B1','binary',0,b.astype(float)),
                ('K1','categorical',0,(level==1).astype(float)),('K1','categorical',1,(level==2).astype(float)),
                *((f'N{i+1}','continuous',0,x) for i,x in enumerate(noise))]
        with tempfile.TemporaryDirectory() as tmp:
            write_study(Path(tmp),groups,[a,y,e,mu0,mu1])
            frame,definitions,labels=orthogonal_tests.study_inputs(tmp,n)
            self.assertEqual(list(frame['K1']),[str(v+1) for v in level])
            self.assertEqual(list(frame['B1']),[str(v+1) for v in b])
            self.assertEqual([d.get('categories_or_unit') for d in definitions[:3]],[None,['1','2'],['1','2','3']])
            np.testing.assert_array_equal(labels['Y'],y)
            selected=orthogonal_tests.run_study(tmp,tmp,n,orthogonal_tests.settings(.1))
            written=json.loads((Path(tmp)/'modifier_tests.json').read_text())
        self.assertIn('M1',selected); self.assertEqual(written['selected'],selected)
        self.assertEqual([r['feature_id'] for r in written['records']],['M1','B1','K1','N1','N2','N3','N4'])
        self.assertEqual(written['records'][2]['grouped_columns'],['K1:level=2','K1:level=3'])

@unittest.skipUnless(R_AVAILABLE,'Rscript with glmnet and jsonlite is not installed')
class REngine(unittest.TestCase):
    def test_engine_checks(self):
        done=subprocess.run(['Rscript',str(ROOT/'tests/test_sample_size_calculator_engine.R'),
            str(CALCULATOR/'binary_selection_methods.R'),str(CALCULATOR/'engine.R'),str(CALCULATOR/'configs/pilot.json')],
            capture_output=True,text=True,env=dict(os.environ,**THREADS))
        self.assertEqual(done.returncode,0,done.stderr)
        self.assertEqual(done.stdout.count('PASS:'),4,done.stdout)
    def pilot(self,tmp,**changes):
        config=Path(tmp)/'config.json'; out=Path(tmp)/'pilot'
        config.write_text(json.dumps(dict(json.loads((CALCULATOR/'configs/pilot-v2.json').read_text()),**changes)))
        done=subprocess.run([str(ROOT/'run_sample_size_calculator.sh'),'--config',str(config),'--output',str(out)],
            capture_output=True,text=True,env=dict(os.environ,PYTHON=sys.executable,**THREADS))
        return done,out
    def test_launcher_runs_a_pilot_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            done,out=self.pilot(tmp)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr+(out/'execution.log').read_text())
            self.assertEqual(json.loads((out/'progress.json').read_text())['state'],'complete')
            result=json.loads((out/'result.json').read_text())
            self.assertIsNone(result['candidate_n']); self.assertIsNone(result['recommended_n'])
            self.assertEqual([(c['n'],c['replications'],c['failures']) for c in result['cells']],[(400,2,0)])
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual(manifest['state'],'complete')
            for name,expected in manifest['outputs'].items():
                self.assertEqual(sha(out/name),expected)
            self.assertTrue((out/'n400_r001/data.csv').is_file())
    @unittest.skipUnless(orthogonal_tests is not None,'the OCI environment is not installed in this interpreter')
    def test_orthogonal_tests_selector_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            done,out=self.pilot(tmp,selector='orthogonal_tests',modifier_q=.2)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr+(out/'execution.log').read_text()+(out/'orthogonal_tests.log').read_text())
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['method'],'Stage 2 orthogonal tests (Cauchy combination, BH)')
            self.assertEqual([(c['n'],c['replications'],c['failures']) for c in result['cells']],[(400,2,0)])
            for record in json.loads((out/'replications.json').read_text()):
                tests=json.loads((out/f'n400_r{record["replicate"]:03d}/modifier_tests.json').read_text())
                self.assertEqual(sorted(tests['selected']),record['selected'])
                self.assertEqual(len(tests['records']),sum(sum(row.values()) for row in result['config']['variables'].values()))
            self.assertEqual(json.loads((out/'orthogonal_tests_worker.json').read_text())['settings']['q'],.2)

if __name__=='__main__': unittest.main()
