"""Freeze matched broad/compact forests and parallel compact production fits."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import time
import common as c


def inputs():
    import pandas as pd
    root=c.n.INPUTS/'inputs'
    return (pd.read_pickle(root/'training.pkl'),pd.read_pickle(root/'heldout.pkl'),
            pd.read_parquet(root/'training_labels.parquet'),pd.read_parquet(root/'heldout_labels.parquet'),
            c.n.read(root/'split.json'))


def production(seed):
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from oci.inference.plain_handoff_stage2_analysis import estimate_outer_fold
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    train,test,tl,vl,split=inputs()
    definitions=c.n.read(c.HERE/'selected_definitions.json')['features']
    policy=c.n.read(c.HERE/f'input_manifest_{c.DATE}.json')['evaluation_plan']
    assert train._oci_row_id.tolist()==tl._oci_row_id.tolist()==split['fit_row_ids']
    assert test._oci_row_id.tolist()==vl._oci_row_id.tolist()==split['heldout_row_ids']
    data=pd.concat([tl,vl]).sort_values('_oci_row_id')[['_oci_row_id','treatment','outcome']].reset_index(drop=True)
    assert data._oci_row_id.tolist()==list(range(1000))
    data['analysis_unit']=data._oci_row_id
    start=time.monotonic()
    with threadpool_limits(limits=1):
        diag=estimate_outer_fold(dataset=data,extracted_fit=train,extracted_heldout=test,definitions=definitions,
            split=split,unit_id_column='analysis_unit',treatment_column='treatment',outcome_column='outcome',
            outcome_type='binary',inner_folds=5,seed=seed,propensity_clip=policy['propensity_clip'],
            min_propensity=policy['min_propensity'],max_propensity=policy['max_propensity'],
            estimation_trees=policy['trees'],output_dir=c.HERE/'production'/f'seed_{seed}')
    assert diag['nuisance_model_family']=='elastic_net'
    return {'seed':seed,'seconds':time.monotonic()-start,'fit_n':diag['effect_fit_rows'],'test_n':diag['effect_estimation_rows']}


def main():
    import numpy as np
    from threadpoolctl import threadpool_limits
    from econml.grf import CausalForest
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder
    c.assert_pre_oracle()
    manifest=c.n.read(c.HERE/f'input_manifest_{c.DATE}.json')
    selection=c.n.read(c.HERE/f'selection_frozen_{c.DATE}.json')
    assert selection['input_manifest_sha256']==c.n.sha(c.HERE/f'input_manifest_{c.DATE}.json')
    c.n.verify(manifest['sources']);c.n.verify(manifest['input_files']);c.n.verify(selection['files'])
    freeze=c.HERE/f'predictions_frozen_{c.DATE}.json'
    if freeze.exists():c.n.verify(c.n.read(freeze)['files']);return
    plan=manifest['evaluation_plan']
    identity={'source_sha256':c.n.sha(__file__),'selection_freeze_sha256':c.n.sha(c.HERE/f'selection_frozen_{c.DATE}.json'),
              'plan':plan,'parallel_production_workers':3,'broad_selection_sha256':c.n.sha(c.BASE/'selected_definitions.json')}
    input_path=c.HERE/'fit_input.json'
    if input_path.exists():assert c.n.read(input_path)==identity,'Fitting procedure changed'
    else:c.n.write(input_path,identity)
    train,test,tl,vl,split=inputs()
    tr=tl.effect_eligible.to_numpy(bool);te=vl.effect_eligible.to_numpy(bool)
    assert tr.sum()==720 and te.sum()==180
    assert train._oci_row_id.tolist()==tl._oci_row_id.tolist()==split['fit_row_ids']
    assert test._oci_row_id.tolist()==vl._oci_row_id.tolist()==split['heldout_row_ids']
    all_files=[input_path,Path(__file__)]
    with threadpool_limits(limits=1):
        for method,definitions in [('matched_residuals',c.n.read(c.HERE/'selected_definitions.json')['features']),
                                  ('broad_matched',c.n.read(c.BASE/'selected_definitions.json')['features'])]:
            modifiers=[d for d in definitions if 'effect_modifier' in d['roles']]
            encoder=_FeatureEncoder(modifiers).fit(train.loc[tr].reset_index(drop=True))
            x=encoder.transform(train.loc[tr].reset_index(drop=True));xv=encoder.transform(test.loc[te].reset_index(drop=True))
            if x.shape[1]==0:x,xv=np.ones((tr.sum(),1)),np.ones((te.sum(),1))
            tres=(tl.treatment-tl.propensity).to_numpy()[tr];yres=(tl.outcome-tl.outcome_prediction).to_numpy()[tr]
            constant=float(tres@yres/(tres@tres))
            for seed in plan['matched_forest_seeds']:
                dest=c.HERE/method/f'seed_{seed}';dest.mkdir(parents=True,exist_ok=True)
                params=dict(n_estimators=200,max_depth=None,min_samples_leaf=10,max_features='sqrt',honest=True,
                            inference=True,subforest_size=4,n_jobs=1,random_state=seed)
                old=c.n.read(c.n.SOURCE/f'refresh/outer_001/comparison_forests/all_candidates/seed_{seed}/metrics.json')
                assert params==old['forest_parameters']
                model=CausalForest(**params).fit(x,tres,yres)
                cate,lo,hi=model.predict(xv,interval=True)
                pred=vl.loc[te].reset_index(drop=True).copy()
                pred['estimated_cate'],pred['lower_95'],pred['upper_95']=cate.ravel(),lo.ravel(),hi.ravel()
                assert np.isfinite(pred[['estimated_cate','lower_95','upper_95']]).all().all()
                pred.to_csv(dest/'predictions.csv',index=False)
                t=pred.treatment-pred.propensity;y=pred.outcome-pred.outcome_prediction
                loss=float(np.mean((y-t*pred.estimated_cate)**2));null=float(np.mean((y-t*constant)**2))
                c.n.write(dest/'metrics.json',{'seed':seed,'parameters':params,'effective_parameters':model.get_params(),
                    'modifiers':len(modifiers),'encoded_columns':x.shape[1],'fit_rows':720,'test_rows':180,
                    'constant_effect':constant,'r_loss':loss,'r_score':1-loss/null,
                    'train_design_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                    'test_design_sha256':hashlib.sha256(xv.tobytes()).hexdigest()})
                all_files.extend([dest/'predictions.csv',dest/'metrics.json'])
                print(json.dumps({'phase':'matched_fit_complete','method':method,'seed':seed,'r_loss':loss}),flush=True)
    c.n.write(c.HERE/'status.json',{'phase':'parallel_compact_production_fits','updated_at':c.n.now(),'seeds':plan['estimation_seeds']})
    with concurrent.futures.ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(production,seed) for seed in plan['estimation_seeds']]
        for future in concurrent.futures.as_completed(futures):
            result=future.result()
            print(json.dumps({'phase':'production_fit_complete',**result}),flush=True)
            dest=c.HERE/'production'/f"seed_{result['seed']}"
            all_files.extend(p for p in dest.glob('*') if p.is_file())
    c.n.verify(manifest['sources']);c.n.verify(manifest['input_files']);c.n.verify(selection['files'])
    c.n.write(freeze,{'frozen_at':c.n.now(),'files':{str(p.resolve()):c.n.sha(p) for p in all_files},
                     'selection_freeze_sha256':c.n.sha(c.HERE/f'selection_frozen_{c.DATE}.json'),
                     'production_fits':3,'matched_residual_fits':3,'broad_matched_fits':3,'oracle_read_by_fit':False})
    c.n.write(c.HERE/'status.json',{'phase':'predictions_frozen','updated_at':c.n.now()})


if __name__=='__main__':main()
