"""Choose a ranked compact modifier prefix using saved inner-fold residuals."""
from pathlib import Path
import json
import math
import common as c


def choose_size(losses):
    import numpy as np
    sizes=sorted(losses)
    arrays={k:np.asarray(losses[k],dtype=float) for k in sizes}
    assert all(a.shape==(5,) and np.isfinite(a).all() for a in arrays.values())
    best=min(sizes,key=lambda k:(float(arrays[k].mean()),k))
    details={}
    for k in sizes:
        difference=arrays[k]-arrays[best]
        excess=float(difference.mean())
        se=float(difference.std(ddof=1)/math.sqrt(5))
        details[str(k)]={'mean_r_loss':float(arrays[k].mean()),'mean_excess_vs_best':excess,
                         'paired_standard_error':se,'within_one_se':bool(excess<=se+1e-12)}
    chosen=min(k for k in sizes if details[str(k)]['within_one_se'])
    return {'best_mean_size':best,'chosen_size':chosen,'options':details,
            'rule':'smallest size within one paired standard error of the minimum mean fold R-loss',
            'formal_error_guarantee':False}


def main():
    import numpy as np
    import pandas as pd
    from threadpoolctl import threadpool_limits
    from econml.grf import CausalForest
    from oci.inference.plain_handoff_stage2_analysis import _FeatureEncoder

    c.assert_pre_oracle()
    upstream=c.verify_upstream()
    complete=c.n.read(c.HERE/'global_review_complete.json')
    active_input=Path(complete.get('input_path',c.HERE/'global_review_input.json'))
    assert complete['input_sha256']==c.n.sha(active_input)
    active_runner=Path(complete.get('runner_path',c.HERE/f'global_review_{c.DATE}.py'))
    assert c.n.read(active_input)['source_sha256']==c.n.sha(active_runner)
    assert complete['response_sha256']==c.n.sha(c.HERE/'global_response.json')
    response=c.n.read(c.HERE/'global_response.json')
    definitions=c.n.read(c.n.INPUTS/'inputs/definitions.json')
    by_id={d['feature_id']:d for d in definitions}
    ranked=[c.full_id(x['id']) for x in response['modifier_ranking']]
    broad=[d['feature_id'] for d in c.n.read(c.BASE/'selected_definitions.json')['features'] if 'effect_modifier' in d['roles']]
    sizes=sorted({0,len(ranked),*(k for k in [8,16,32] if k<=len(ranked))})
    subsets={str(k):ranked[:k] for k in sizes}
    subsets['broad_224']=broad
    params={'n_estimators':200,'min_samples_leaf':10,'max_features':'sqrt','honest':True,
            'inference':True,'subforest_size':4,'n_jobs':1,'max_samples':0.45}
    identity={'source_sha256':c.n.sha(__file__),'common_sha256':c.n.sha(c.HERE/'common.py'),
              'global_review_complete_sha256':c.n.sha(c.HERE/'global_review_complete.json'),
              'numerical_freeze_sha256':c.n.sha(c.BASE/f'numerical_frozen_{c.DATE}.json'),
              'subsets':subsets,'forest_parameters':params,'seed_bases':[120042,1120042],
              'seed_formula':'base + inner_fold_position * 10000','propensity_bounds':[0.1,0.9]}
    input_path=c.HERE/'size_selection_input.json'
    if input_path.exists():assert c.n.read(input_path)==identity,'Size-selection inputs changed'
    else:c.n.write(input_path,identity)
    fingerprint=c.n.sha(input_path)
    train=pd.read_pickle(c.n.INPUTS/'inputs/training.pkl').set_index('_oci_row_id',drop=False)
    labels=pd.read_parquet(c.n.INPUTS/'inputs/training_labels.parquet').set_index('_oci_row_id',drop=False)
    split=c.n.read(c.n.INPUTS/'inputs/split.json')
    assert set(train.index)==set(labels.index)==set(split['fit_row_ids'])
    records=[]
    with threadpool_limits(limits=1):
        for position,fold in enumerate(split['inner_splits'],1):
            fold_number=int(fold.get('inner_fold',position))
            paths=list((c.BASE/'numerical').glob(f'*/fold_{fold_number:03d}/nuisances.json'))
            assert len(paths)==1
            nuisance=c.n.read(paths[0])['result']
            ids,valid_ids=fold['fit_row_ids'],fold['heldout_row_ids']
            ft,fv=train.loc[ids].reset_index(drop=True),train.loc[valid_ids].reset_index(drop=True)
            t,y=labels.loc[ids,['treatment','outcome']].to_numpy(float).T
            tv,yv=labels.loc[valid_ids,['treatment','outcome']].to_numpy(float).T
            e,m,ev,mv=[np.asarray(nuisance[k],float) for k in ['training_propensity','training_outcome','validation_propensity','validation_outcome']]
            assert e.shape==m.shape==t.shape and ev.shape==mv.shape==tv.shape
            keep=(e>=.1)&(e<=.9);keepv=(ev>=.1)&(ev<=.9)
            tr,yr=(t-e)[keep],(y-m)[keep]
            tvr,yvr=(tv-ev)[keepv],(yv-mv)[keepv]
            constant=float(tr@yr/(tr@tr))
            for label,keys in subsets.items():
                encoder=_FeatureEncoder([by_id[k] for k in keys]).fit(ft.loc[keep].reset_index(drop=True))
                x=encoder.transform(ft.loc[keep].reset_index(drop=True))
                xv=encoder.transform(fv.loc[keepv].reset_index(drop=True))
                for seed_base in identity['seed_bases']:
                    seed=seed_base+position*10000
                    path=c.HERE/'size_validation'/f'fold_{fold_number:03d}'/f'size_{label}'/f'seed_{seed}.json'
                    if path.exists():
                        record=c.n.read(path)
                        assert record['input_sha256']==fingerprint
                        assert record['feature_ids']==keys
                    else:
                        if keys:
                            pred=CausalForest(**params,random_state=seed).fit(x,tr,yr).predict(xv).reshape(-1)
                        else:pred=np.full(len(tvr),constant)
                        assert np.isfinite(pred).all()
                        errors=(yvr-tvr*pred)**2
                        record={'input_sha256':fingerprint,'inner_fold':fold_number,'size':label,'seed':seed,
                                'feature_ids':keys,'encoded_columns':x.shape[1],'fit_n':int(keep.sum()),
                                'validation_n':int(keepv.sum()),'validation_row_ids':np.asarray(valid_ids)[keepv].astype(int).tolist(),
                                'constant_effect':constant,'predictions':pred.tolist(),'squared_errors':errors.tolist(),
                                'r_loss':float(errors.mean())}
                        c.n.write(path,record)
                    records.append(record)
                c.n.write(c.HERE/'status.json',{'phase':'compact_size_validation','inner_fold':fold_number,
                                              'size':label,'updated_at':c.n.now()})
                print(json.dumps({'fold':fold_number,'size':label,'mean_r_loss':float(np.mean([r['r_loss'] for r in records[-2:]]))}),flush=True)
    fold_losses={label:[float(np.mean([r['r_loss'] for r in records if r['size']==label and r['inner_fold']==int(f.get('inner_fold',i))])) for i,f in enumerate(split['inner_splits'],1)] for label in subsets}
    choice=choose_size({k:fold_losses[str(k)] for k in sizes})
    choice.update(fold_losses=fold_losses,input_sha256=fingerprint,chosen_at=c.n.now(),broad_reference_mean=float(np.mean(fold_losses['broad_224'])))
    c.n.write(c.HERE/'size_choice.json',choice)

    selected_modifiers=set(ranked[:choice['chosen_size']])
    confounders={c.full_id(e['id']):e for e in response['confounders']}
    modifiers={c.full_id(e['id']):e for e in response['modifier_ranking']}
    group_by_member={c.full_id(k):g for g in response['groups'] for k in g['members']}
    first={d['feature_id']:d for d in c.n.read(c.BASE/'role_report.json')['decisions']}
    decisions,selected=[],[]
    for definition in definitions:
        key=definition['feature_id'];roles=[];explanations=[];entries=[]
        if key in confounders:
            roles.append('confounder');entries.append(confounders[key]);explanations.append(confounders[key]['rationale'])
        if key in selected_modifiers:
            roles.append('effect_modifier');entries.append(modifiers[key]);explanations.append(modifiers[key]['rationale'])
        elif key in modifiers:
            explanations.append(f"Modifier rank {ranked.index(key)+1} exceeds the training-selected prefix of {choice['chosen_size']}.")
        if not entries:explanations.append('Not retained in the final roles after global consolidation and compact-size selection.')
        citations=sorted({c.evidence_id(key.removeprefix(c.PREFIX),ref) for entry in entries for ref in entry['evidence_refs']})
        stability=('consistent' if all(e['stability']=='consistent' for e in entries) else 'mixed') if entries else first[key]['stability']
        decision={'feature_id':key,'roles':roles,'evidence_ids':citations,'stability':stability,
                  'rationale':' '.join(explanations),'group_id':group_by_member[key]['group_id'],
                  'consolidation_rationale':group_by_member[key]['consolidation_rationale'],
                  'first_pass_roles':first[key]['roles'],'modifier_rank':ranked.index(key)+1 if key in modifiers else None}
        decisions.append(decision)
        if roles:selected.append({**definition,'roles':roles})
    c.n.write(c.HERE/'selected_definitions.json',{'features':selected})
    c.n.write(c.HERE/'role_report.json',{'status':'complete','decisions':decisions,'summary':response['summary'],
                                      'themes':response['groups'],'size_choice':choice})
    new_manifest={**upstream,'consolidation':{'parent_manifest_sha256':c.n.sha(c.BASE/f'input_manifest_{c.DATE}.json'),
                                          'protocol_sha256':c.n.sha(c.HERE/f'PROTOCOL_{c.DATE}.md'),
                                          'max_ranked_modifiers':32,'chosen_modifiers':choice['chosen_size']}}
    new_manifest['sources']={**upstream['sources'],**{str(p.resolve()):c.n.sha(p) for p in [Path(__file__),c.HERE/'common.py',c.HERE/f'global_review_{c.DATE}.py',active_runner]}}
    c.n.write(c.HERE/f'input_manifest_{c.DATE}.json',new_manifest)
    files=[c.HERE/name for name in ['global_prompt.json','global_response.json','global_review_input.json','global_review_complete.json',
                                   'size_selection_input.json','size_choice.json','role_report.json','selected_definitions.json',f'PROTOCOL_{c.DATE}.md']]
    files.extend([active_input,Path(complete.get('prompt_path',c.HERE/'global_prompt.json'))])
    if (c.HERE/'PROTOCOL_AMENDMENT_revision2_2026-09-21.md').exists():
        files.extend([c.HERE/'PROTOCOL_AMENDMENT_revision2_2026-09-21.md',c.HERE/'global_review_transition_revision2.json'])
    files.extend(sorted((c.HERE/'size_validation').glob('**/*.json')))
    c.n.write(c.HERE/f'selection_frozen_{c.DATE}.json',{'frozen_at':c.n.now(),'files':{str(p.resolve()):c.n.sha(p) for p in files},
                'input_manifest_sha256':c.n.sha(c.HERE/f'input_manifest_{c.DATE}.json'),
                'candidate_count':352,'selected_count':len(selected),
                'roles':{role:sum(role in d['roles'] for d in selected) for role in ['confounder','effect_modifier']},
                'oracle_or_outer_test_information_read':False})
    c.n.write(c.HERE/'status.json',{'phase':'consolidated_selection_frozen','updated_at':c.n.now(),
                                 'chosen_modifiers':choice['chosen_size'],'confounders':len(confounders)})
    print(json.dumps({'choice':choice,'selected_count':len(selected),'confounders':len(confounders)}),flush=True)


if __name__=='__main__':main()
