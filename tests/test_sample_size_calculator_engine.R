# Deterministic mathematical/generator checks of sample_size_calculator/engine.R; not a power study.
# Usage: Rscript tests/test_sample_size_calculator_engine.R METHODS.R ENGINE.R CONFIG.json
args <- commandArgs(trailingOnly=TRUE)
source(args[1]); source(args[2]); cfg <- fromJSON(args[3]); groups <- make_groups(cfg)
cal <- calibrate(cfg,groups); pop <- covariates(30000,cfg,groups,cfg$calibration_seed)
p <- probabilities(cfg,pop,cal)
stopifnot(abs(mean(p$e)-cfg$treated_fraction)<1e-10,
          abs(mean(p$mu0)-cfg$baseline_risk)<1e-10)
truth <- cfg$constant_effect+as.numeric(pop$effect %*% cal$weights$beta)
stopifnot(max(abs(p$truth_logor-truth))<1e-10)
stopifnot(nrow(groups)==110,ncol(pop$X)==111,
          nrow(pop$map[pop$map$kind=='categorical',])==2)
zero_a <- cfg; zero_a$treatment_coefficient <- 0
ca <- calibrate(zero_a,groups); pa <- probabilities(zero_a,pop,ca)
stopifnot(max(abs(pa$e-cfg$treated_fraction))<1e-10,sd(p$e)>.01,
          max(abs(pa$mu0-p$mu0))<1e-12)
zero_g <- cfg; zero_g$outcome_coefficient <- 0
cg <- calibrate(zero_g,groups); pg <- probabilities(zero_g,pop,cg)
stopifnot(max(abs(pg$mu0-cfg$baseline_risk))<1e-10,
          max(abs(pg$e-p$e))<1e-12)
rd <- cfg; rd$scale <- 'RD'; rd$modifier_strength <- 1
stopifnot(inherits(try(calibrate(rd,groups),silent=TRUE),'try-error'))
# Reduced valid RD design, all null, and group representation.
rd$variables <- lapply(rd$variables,function(v) list(continuous=0,binary=0,categorical=0))
rd$variables$confounder$continuous <- 1; rd$variables$modifier$continuous <- 1
rd$variables$noise$continuous <- 10; rd$treatment_coefficient <- .2; rd$outcome_coefficient <- .2
rd$modifier_strength <- .05; rd$constant_effect <- .02; rd$baseline_risk <- .4
rg <- make_groups(rd); rc <- calibrate(rd,rg); rp <- covariates(1000,rd,rg,100)
pr <- probabilities(rd,rp,rc)
stopifnot(max(abs(pr$truth_rd-(.02+as.numeric(rp$effect %*% rc$weights$beta))))<1e-14)
null <- cfg; null$modifier_strength <- 0; nc <- calibrate(null,groups)
np <- probabilities(null,pop,nc); stopifnot(max(abs(np$truth_logor-null$constant_effect))<1e-10)
# Both-role variables have all three terms.
both <- cfg; both$variables$both$binary <- 1; bg <- make_groups(both); bw <- weights(both,bg)
ix <- bg$role=='both'; stopifnot(all(bw$alpha[ix]!=0),all(bw$gamma[ix]!=0),all(bw$beta[ix]!=0))
cat('PASS: calibration, native logOR/RD identities, separate confounding controls, group encoding, both-role terms, null truth, invalid RD rejection.\n')
# Exact-support decision logic is exercised using constructed records, not new simulations.
sc <- cfg; sc$mode <- 'grid'; sc$replications <- 200; sc$target_probability <- .8
sc$sample_sizes <- c(400,800); fake <- list()
truth_groups <- groups$group[abs(cal$weights$beta)>0]
for(n in sc$sample_sizes) for(j in 1:200) fake[[length(fake)+1]] <- list(n=n,replicate=j,
  selected=truth_groups,truth=truth_groups,tp=5,fp=0,success=TRUE,exact=TRUE,all_recovered=TRUE,error=NULL)
answer <- summarize_records(fake,sc,groups,cal)
stopifnot(answer$candidate_n==400,is.null(answer$recommended_n))
fake[[201]]$error <- 'deliberate fit failure'; fake[[201]]$success <- FALSE
blocked <- summarize_records(fake,sc,groups,cal)
stopifnot(is.null(blocked$candidate_n),blocked$cells[[2]]$failures==1)
sc$mode <- 'pilot'; no_recommendation <- summarize_records(fake,sc,groups,cal)
stopifnot(is.null(no_recommendation$candidate_n),is.null(no_recommendation$recommended_n))
cat('PASS: synthetic positive grid crossing, later-failure blocking, pilot suppression.\n')
# v2 operating event at original-variable level; no implicit integer FP threshold.
v2 <- cfg; v2$schema_version <- 2; v2$sensitivity <- .5; v2$false_positive_rate <- .25
all <- c('M1','M2','C1','N1','N2','N3'); truth <- c('M1','M2')
a <- detection_metrics(c('M1','N1'),truth,all,v2)
stopifnot(a$success,a$sensitivity==.5,a$false_positive_rate==.25,a$specificity==.75)
stopifnot(!detection_metrics(c('M1','N1','N2'),truth,all,v2)$success)
# Empty denominators are undefined, constraints are vacuous.
null <- detection_metrics(character(),character(),all,v2)
stopifnot(null$success,is.null(null$sensitivity),null$specificity==1)
only <- detection_metrics('M1','M1','M1',v2)
stopifnot(only$success,is.null(only$false_positive_rate),is.null(only$specificity))
cat('PASS: v2 sensitivity/FPR event and empty denominators.\n')
# Study seeds: no two studies in a full grid share a seed, counting each study's
# offsets (covariates, outcomes, nuisance folds, multipliers), nor the calibration seed.
grid <- cfg; grid$replications <- 2500
grid$sample_sizes <- c(200,400,5000,10200,10400,15000,19999,20000)
offsets <- c(0:3,2+as.vector(outer(100*(1:2),1:3,'+')))
seeds <- unlist(lapply(grid$sample_sizes,function(n) vapply(seq_len(2500),function(r) study_seed(grid,n,r),integer(1))))
used <- as.vector(outer(seeds,offsets,'+'))
stopifnot(!anyDuplicated(used),!(grid$calibration_seed %in% used),
          study_seed(grid,400,2)!=study_seed(grid,10400,1),max(used)<.Machine$integer.max)
cat('PASS: study seeds are unique across a full grid.\n')
