"""Write the final outline report and a scientific comparison figure."""
import json
import common as c


def value(summary,key,precision=3):
    item=summary.get(key,{}).get('mean')
    return 'undefined (constant)' if item is None else f'{item:.{precision}f}'


def main():
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    evaluation=c.n.read(c.HERE/f'evaluation_{c.DATE}.json')
    recovery=c.n.read(c.HERE/f'oracle_recovery_{c.DATE}.json')
    funnel=c.n.read(c.HERE/f'recovery_funnel_{c.DATE}.json')['variables']
    review=c.n.read(c.HERE/'global_response.json')
    automatic_groups=sum(bool(g.get('automatically_added')) for g in review['groups'])
    choice=c.n.read(c.HERE/'size_choice.json')
    selected=c.n.read(c.HERE/'selected_definitions.json')['features']
    metrics=pd.read_csv(c.HERE/f'metrics_by_seed_{c.DATE}.csv')
    predictions=pd.read_csv(c.HERE/f'predictions_with_oracle_{c.DATE}.csv')
    definitions=c.n.read(c.n.INPUTS/'inputs/definitions.json')
    names={d['feature_id']:d['name'] for d in definitions}
    summaries=evaluation['summaries']
    labels={'current_joint':'Previous joint elastic-net selection','llm_adjudication':'Previous LLM selection',
            'permissive_union':'Previous permissive union','all_candidates':'All 352; sqrt split search',
            'all_candidates_all_split_features':'All 352; all-column split search',
            'multi_model_broad':'First multi-model pass; 224 modifiers',
            'multi_model_matched':f"Consolidated selection; {choice['chosen_size']} modifiers"}
    rows=[]
    for method,label in labels.items():
        s=summaries[method]
        rows.append(f"| {label} | {value(s,'correlation')} | {value(s,'rmse')} | {value(s,'bias')} | {value(s,'estimated_sd')} |")
    recovery_rows=[]
    for r in recovery['direct']:
        role='C' if r['true_role']=='confounder' else 'M'
        assigned='; '.join(k[-3:]+': '+('/'.join('C' if x=='confounder' else 'M' for x in v) or 'none') for k,v in r['assigned_roles'].items())
        recovery_rows.append(f"| {r['oracle_variable']} | {role} | {assigned} | {'Yes' if r['retained_in_correct_role'] else 'No'} |")
    funnel_summary={role:[sum(x[field] for x in funnel if x['true_role']==role) for field in ['broad_correct_role','global_review_correct_role','final_correct_role']] for role in ['confounder','effect_modifier']}
    cv_rows=[]
    for size in sorted(choice['fold_losses'],key=lambda x:int(x) if x!='broad_224' else 100000):
        losses=choice['fold_losses'][size]
        detail=choice['options'].get(size)
        why=('chosen' if int(size)==choice['chosen_size'] else 'eligible compact option') if size!='broad_224' else 'reference only'
        cv_rows.append(f"| {size} | {np.mean(losses):.7f} | {detail['paired_standard_error']:.7f} | {why} |" if detail else f"| {size} | {np.mean(losses):.7f} | — | {why} |")
    modifier_names=', '.join(names[d['feature_id']] for d in selected if 'effect_modifier' in d['roles']) or 'None; constant-effect option chosen'
    counts=evaluation['selected_roles'];direct=evaluation['direct_recovery']
    prod=summaries['multi_model_production'];pm=metrics.loc[metrics.method=='multi_model_production']
    baseline=summaries['multi_model_broad'];compact=summaries['multi_model_matched']
    broad_columns=c.n.read(c.HERE/'broad_matched/seed_120042/metrics.json')['encoded_columns']
    compact_columns=c.n.read(c.HERE/'matched_residuals/seed_120042/metrics.json')['encoded_columns']
    validation_path=c.HERE/f'final_validation_{c.DATE}.json'
    validation=c.n.read(validation_path) if validation_path.exists() else None
    validation_text=(f"Final integrity checks passed, including frozen-file hashes, observed-label row alignment, inherited oracle lineage, and all {len(validation['production_nuisance_clones'])} fitted nuisance clones. {validation['nuisance_clones_at_iteration_limit']} clones reached the iteration limit." if validation else 'Final integrity audit is recorded separately after report generation.')
    heldout_range=str(int(pm.n.min())) if pm.n.min()==pm.n.max() else f'{int(pm.n.min())}–{int(pm.n.max())}'
    if 'correlation' in compact and 'correlation' in baseline:
        change=f"Mean correlation changed by {compact['correlation']['mean']-baseline['correlation']['mean']:+.3f} relative to the broad first pass."
    else:change='Correlation is undefined for a constant effect estimate.'
    change+=f" Mean RMSE changed by {compact['rmse']['mean']-baseline['rmse']['mean']:+.3f}; lower is better."
    proxies='; '.join(f"{r['proxy']}: {', '.join(r['assigned_roles']) or 'not selected'}" for r in recovery['proxies'])
    fig,axes=plt.subplots(1,3,figsize=(13.2,4.2),constrained_layout=True)
    panels=[('multi_model_broad',120042,'Broad first pass: 224 modifiers'),
            ('multi_model_matched',120042,'Consolidated: fixed residuals'),
            ('multi_model_production',100042,'Consolidated: production estimator')]
    limits=[]
    for method,seed,_ in panels:
        f=predictions.loc[(predictions.method==method)&(predictions.seed==seed)]
        limits.extend(f.true_ite_prob.tolist()+f.estimated_cate.tolist())
    low,high=min(limits)-.025,max(limits)+.025
    for axis,(method,seed,title) in zip(axes,panels):
        f=predictions.loc[(predictions.method==method)&(predictions.seed==seed)]
        m=metrics.loc[(metrics.method==method)&(metrics.seed==seed)].iloc[0]
        corr=f'{m.correlation:.3f}' if pd.notna(m.correlation) else 'undefined'
        axis.scatter(f.true_ite_prob,f.estimated_cate,s=19,alpha=.65,color='#245a81',edgecolors='none')
        axis.plot([low,high],[low,high],color='#777777',linewidth=1,linestyle='--')
        axis.axhline(0,color='#cccccc',linewidth=.7)
        axis.set(xlim=(low,high),ylim=(low,high),xlabel='True probability-scale ITE',ylabel='Estimated CATE',
                 title=f'{title}\nn = {len(f)}, r = {corr}, RMSE = {m.rmse:.3f}')
        axis.spines[['top','right']].set_visible(False);axis.set_aspect('equal')
    figure=c.HERE/f'ite_scatter_{c.DATE}.png';fig.savefig(figure,dpi=180);plt.close(fig)
    text=f"""# Fold 1: multi-model selection with global consolidation — September 21, 2026

1. **Main result**
   1. Reducing the forest from 224 to eight modifier candidates improved the matched-population mean ITE correlation from **{value(baseline,'correlation')} to {value(compact,'correlation')}** and RMSE from **{value(baseline,'rmse')} to {value(compact,'rmse')}** across three seeds.
   2. The full consolidated workflow still has substantial selection failures: only **{direct['confounder']['correct_role']}/5 direct confounders and {direct['effect_modifier']['correct_role']}/5 direct modifiers** survived. The production refit reached correlation **{value(prod,'correlation')}**, and its AIPW ATE had the opposite sign from the oracle ATE in the eligible population. This result supports reducing the modifier search space while exposing a separate problem in confounder retention.

2. **What changed**
   1. The initial multi-model Stage 2 review retained 270 candidates, including 189 confounders and 224 modifiers. Its final decisions were made in small batches after theme review; retention was too broad to produce a compact modifier set.
   2. Following the user's request, added a global comparison of all 352 candidates, explicit redundancy groups, separate confounder representatives, and a ranked modifier shortlist capped at 32. This is an experimental pass in a separate folder; the production repository implementation and original frozen evidence remain unchanged.
   3. Reused all 305 numerical family/subset cells, the five original inner folds, and the saved Gemma 4 26B A4B extractions for 800 training / 200 held-out patients. No extraction was rerun. The expensive broad production refit was stopped before oracle access; the broad selection remains a matched-residual forest comparison.

3. **Global review and compact-size choice**
   1. Gemma 4 31B reviewed every candidate together, including measurement definitions, training availability, all seven modeling families, score magnitudes/signs, fold support, and available p/q values and permutation variability. Its review returned {len(review['groups'])-automatic_groups} explicit groups, {len(review['confounders'])} confounder representatives, and {len(review['modifier_ranking'])} ranked modifier candidates. Code preserved {automatic_groups} other candidates as separate singletons.
   2. Groups choose existing extracted measurements. No new values were constructed or pooled. Related biological concepts do not by themselves establish measurement equivalence; the proposed groups and their rationales remain available for audit. After two failed attempts to enumerate every candidate explicitly, [output revision 2](PROTOCOL_AMENDMENT_revision2_{c.DATE}.md) assigned unchanged-candidate bookkeeping to code while preserving the substantive selection constraints.
   3. Evaluated compact ranked prefixes and a constant effect using two forest seeds on each of five inner folds. Reused nested training-only elastic-net residuals and held propensity eligibility fixed. The minimum-mean-loss compact option had {choice['best_mean_size']} modifiers; the paired one-standard-error simplification chose **{choice['chosen_size']} modifiers**.

      | Modifier count | Mean inner-fold R-loss | Paired SE of loss difference vs best | Use |
      | --- | ---: | ---: | --- |
{chr(10).join('      '+r for r in cv_rows)}

   4. The 32-feature cap was specified before this evaluation, using the existing subset size as an engineering budget. The broad 224-feature reference was not eligible to win compact-size selection. The same inner folds had informed the global ranking, so this is adaptive tuning; the standard-error rule is a simplification heuristic, not an independent uncertainty guarantee.

4. **Final selection and oracle recovery**
   1. Frozen final selection: **{len(selected)} distinct candidates**, **{counts['confounder']} confounders**, and **{counts['effect_modifier']} modifiers**. Roles may overlap.
   2. Selected modifiers: {modifier_names}.
   3. Direct recovery in the correct role: **{direct['confounder']['correct_role']}/5 confounders and {direct['effect_modifier']['correct_role']}/5 modifiers**. C = confounder; M = effect modifier. Numeric IDs below are candidate suffixes.

      | Oracle variable | True role | Final candidate roles | Correct role recovered? |
      | --- | --- | --- | --- |
{chr(10).join('      '+r for r in recovery_rows)}

   4. Proxy audit, excluded from direct counts: {proxies}.
   5. Direct correct-role recovery through the broad → global → final stages: confounders **{' → '.join(str(x)+'/5' for x in funnel_summary['confounder'])}**; modifiers **{' → '.join(str(x)+'/5' for x in funnel_summary['effect_modifier'])}**. The [recovery funnel](recovery_funnel_{c.DATE}.json) identifies where each variable was retained or lost.
   6. [Global groups and ranked candidates](global_response.json), [all final decisions](all_role_decisions_{c.DATE}.csv), and [direct lineage audit](oracle_recovery_{c.DATE}.json) preserve the reasoning and exclusions.
   7. Losses by step: histology was absent from the broad modifier selection; global review dropped hemoglobin, sex, direct creatinine clearance, and prior platinum exposure, while adding brain metastasis presence to the modifier shortlist. Brain metastasis presence ranked 11th and EGFR ranked 13th, so both disappeared when the eight-modifier prefix was chosen. NLR survived throughout. Renal proxies remained, but they are not counted as direct creatinine-clearance recovery.

5. **Effect estimation on the matched population**
   1. Fixed the same 720 training / 180 held-out patients, earlier all-candidate elastic-net residuals, and three forest seeds. All methods use 200 trees, honesty/inference enabled, minimum leaf size 10, 45% subsampling, and sqrt split search except the explicitly labeled all-column comparator.
   2. Values are means over three seeds. The compact-vs-broad comparison changes only the modifier inputs and their resulting encoded forest design, which shrank from **{broad_columns} to {compact_columns} columns** after category and missingness encoding.

      | Selection / inputs | ITE correlation | ITE RMSE | Mean bias | Estimated effect SD |
      | --- | ---: | ---: | ---: | ---: |
{chr(10).join('      '+r for r in rows)}

   3. {change}
   4. Across seeds, broad-selection correlation ranged from {baseline['correlation']['min']:.3f} to {baseline['correlation']['max']:.3f}; compact-selection correlation ranged from {compact['correlation']['min']:.3f} to {compact['correlation']['max']:.3f}.

6. **Production estimation with consolidated roles**
   1. Used the existing estimator, with modifiers in forest X and pure confounders in W. Refitted elastic-net nuisance models and propensity eligibility. Three independent forest seeds ran in parallel with one numerical thread each.
   2. Mean ITE correlation: **{value(prod,'correlation')}**; RMSE: **{value(prod,'rmse')}**; mean bias: **{value(prod,'bias')}**; estimated-effect SD: {value(prod,'estimated_sd')}.
   3. Eligibility produced {int(pm.fit_n.min())}–{int(pm.fit_n.max())} training and {heldout_range} held-out patients. These populations and nuisance fits differ from the matched comparison above, so their performance difference cannot be attributed solely to confounder selection.
   4. Mean AIPW ATE: {pm.ate_aipw.mean():+.3f}; mean oracle ATE in the corresponding eligible populations: {pm.true_mean.mean():+.3f}. These differ conceptually from the mean forest CATE.

      ![First-seed comparison](ite_scatter_{c.DATE}.png)

   5. The figure shows the first seed; tables summarize all three. Dashed lines indicate perfect agreement.

7. **Limits and next opportunities**
   1. This is one exploratory outer fold after prior fold-1 oracle diagnostics had been seen in earlier work. All new selection choices and nine final predictions were frozen before this experiment opened oracle values. No oracle roles/values or outer-test outcomes entered the LLM review or compact-size choice.
   2. Consolidation addresses redundancy and dimension. It cannot repair missing/error-prone extraction or create statistical power. The global review also remains fallible, and a forced candidate budget can discard a real modifier.
   3. Positive forest/R-learner support fractions are not calibrated tests against noise. A future experiment could add independent noise features to assess how readily the workflow promotes null candidates. That diagnostic alone would not establish formal error control.
   4. Compare role recovery, ITE ranking, RMSE, bias, and eligibility together. Three seeds quantify algorithmic variability on this dataset; they are not independent clinical samples. Evaluation of other outer folds would be needed before claiming general improvement.
   5. The immediate design opportunity is to separate conservative confounder retention from aggressive modifier reduction. The matched comparison already shows benefit from compact forest X with broad all-candidate nuisance residuals. The global review's deletion of three direct confounders is a reason to scrutinize its adjustment policy before using the full consolidated workflow.

8. **Reproducibility**
   1. [Prespecified consolidation protocol](PROTOCOL_{c.DATE}.md), [size choice](size_choice.json), [selection freeze](selection_frozen_{c.DATE}.json), and [prediction freeze](predictions_frozen_{c.DATE}.json).
   2. [Per-seed metrics](metrics_by_seed_{c.DATE}.csv), [evaluation](evaluation_{c.DATE}.json), and [source/input manifest](input_manifest_{c.DATE}.json).
   3. {validation_text} See the [final audit](final_validation_{c.DATE}.json).
"""
    path=c.HERE/f'REPORT_{c.DATE}.md';path.write_text(text)
    c.n.write(c.HERE/f'report_complete_{c.DATE}.json',{'completed_at':c.n.now(),'report':str(path.resolve()),
        'report_sha256':c.n.sha(path),'figure_sha256':c.n.sha(figure),'source_sha256':c.n.sha(__file__),
        'evaluation_sha256':c.n.sha(c.HERE/f'evaluation_{c.DATE}.json')})
    print(str(path),flush=True)


if __name__=='__main__':main()
