# Generic binary DGP + adapter to the existing, unchanged research selectors.
# No work is performed when sourced; run.R owns execution.
make_groups <- function(cfg) {
  rows <- list(); index <- 0L
  for (role in c('confounder','modifier','both','noise')) {
    number <- 0L
    for (kind in c('continuous','binary','categorical')) {
      count <- cfg$variables[[role]][[kind]]
      if (count == 0) next
      for (j in seq_len(count)) {
        number <- number+1L; index <- index+1L
        rows[[index]] <- data.frame(group=paste0(switch(role,confounder='C',modifier='M',both='B',noise='N'),number),
          role=role,kind=kind,index=index)
      }
    }
  }
  do.call(rbind,rows)
}

population_row <- function(cfg,group) {
  if(is.null(cfg$population)) return(NULL)
  if(is.data.frame(cfg$population)) {
    i <- match(group,cfg$population$group)
    row <- as.list(cfg$population[i,,drop=FALSE])
    row$probabilities <- unlist(row$probabilities)
    return(row)
  }
  cfg$population[[which(vapply(cfg$population,function(x) x$group==group,logical(1)))]]
}

covariates <- function(n,cfg,groups,seed) {
  set.seed(seed)
  latent <- matrix(rnorm(n*nrow(groups)),n,nrow(groups))
  common <- rnorm(n)
  latent <- sqrt(cfg$correlation)*common + sqrt(1-cfg$correlation)*latent
  columns <- list(); map <- list(); effect <- matrix(0,n,nrow(groups))
  for (g in seq_len(nrow(groups))) {
    z <- latent[,g]; kind <- groups$kind[g]
    row <- population_row(cfg,groups$group[g])
    prevalence <- if(is.null(row)) cfg$binary_prevalence else row$prevalence
    probs <- if(is.null(row)) cfg$category_probs else unlist(row$probabilities)
    if (kind=='continuous') {
      values <- matrix(sqrt(3)*(2*pnorm(z)-1),n,1)
      effect[,g] <- values[,1]
    } else if (kind=='binary') {
      values <- matrix(as.numeric(pnorm(z)>1-prevalence),n,1)
      effect[,g] <- values[,1]
    } else {
      level <- findInterval(pnorm(z),cumsum(probs))+1L
      values <- sapply(2:length(probs),function(k) as.numeric(level==k))
      effect[,g] <- (level-1)/(length(probs)-1)
    }
    for (j in seq_len(ncol(values))) {
      k <- length(columns)+1L; columns[[k]] <- values[,j]
      map[[k]] <- data.frame(coordinate=sprintf('x%03d',k-1L),group=groups$group[g],
        role=groups$role[g],kind=kind,contrast=j)
    }
  }
  X <- do.call(cbind,columns); map <- do.call(rbind,map); colnames(X) <- map$coordinate
  list(X=X,map=map,effect=effect)
}

weights <- function(cfg,groups) {
  if(!is.null(cfg$population)) {
    get <- function(key) vapply(groups$group,function(g) as.numeric(population_row(cfg,g)[[key]]),numeric(1))
    return(list(alpha=get('alpha'),gamma=get('gamma'),beta=get('beta')))
  }
  adjustment <- groups$role %in% c('confounder','both')
  modifier <- groups$role %in% c('modifier','both')
  sign <- rep(0,nrow(groups)); sign[modifier] <- rep(c(1,-1),length.out=sum(modifier))
  list(alpha=as.numeric(adjustment)*cfg$treatment_coefficient,
    gamma=as.numeric(adjustment)*cfg$outcome_coefficient,
    beta=sign*cfg$modifier_strength)
}

calibrate <- function(cfg,groups) {
  population <- covariates(30000,cfg,groups,cfg$calibration_seed)
  w <- weights(cfg,groups)
  root <- function(score,target) uniroot(function(b) mean(plogis(b+score))-target,c(-100,100),tol=1e-12)$root
  a0 <- root(as.numeric(population$effect %*% w$alpha),cfg$treated_fraction)
  g0 <- root(as.numeric(population$effect %*% w$gamma),cfg$baseline_risk)
  if(cfg$scale=='RD') {
    lower <- ifelse(groups$kind=='continuous',-sqrt(3),0)
    upper <- ifelse(groups$kind=='continuous',sqrt(3),1)
    bound <- function(co) c(sum(pmin(co*lower,co*upper)),sum(pmax(co*lower,co*upper)))
    baseline <- plogis(g0+bound(w$gamma)); delta <- cfg$constant_effect+bound(w$beta)
    if(baseline[1]+delta[1]<=0 || baseline[2]+delta[2]>=1)
      stop('RD inputs cannot guarantee valid probabilities over the covariate support. Reduce modifier/baseline associations or change baseline risk. No clipping was applied.')
  }
  list(treatment_intercept=a0,outcome_intercept=g0,calibration_n=30000,
    calibration_seed=cfg$calibration_seed,weights=w)
}

probabilities <- function(cfg,pop,cal) {
  e <- plogis(cal$treatment_intercept+as.numeric(pop$effect %*% cal$weights$alpha))
  eta0 <- cal$outcome_intercept+as.numeric(pop$effect %*% cal$weights$gamma)
  effect <- cfg$constant_effect+as.numeric(pop$effect %*% cal$weights$beta)
  mu0 <- plogis(eta0)
  mu1 <- if(cfg$scale=='logOR') plogis(eta0+effect) else mu0+effect
  if(any(!is.finite(c(e,mu0,mu1))) || min(c(e,mu0,mu1))<=0 || max(c(e,mu0,mu1))>=1)
    stop('Non-finite or boundary probabilities; revise the input mechanism.')
  data.frame(e=e,mu0=mu0,mu1=mu1,truth_rd=mu1-mu0,truth_logor=qlogis(mu1)-qlogis(mu0))
}

detection_metrics <- function(selected,truth,all_groups,cfg) {
  tp <- length(intersect(selected,truth)); fp <- length(setdiff(selected,truth))
  positives <- length(truth); negatives <- length(setdiff(all_groups,truth))
  sensitivity <- if(positives==0) NULL else tp/positives
  fpr <- if(negatives==0) NULL else fp/negatives
  if(!is.null(cfg$schema_version) && cfg$schema_version==2) {
    success <- (is.null(sensitivity) || sensitivity+1e-12>=cfg$sensitivity) &&
      (is.null(fpr) || fpr<=cfg$false_positive_rate+1e-12)
  } else {
    need <- if(cfg$goal=='all') positives else ceiling(cfg$recovery_fraction*positives-1e-12)
    success <- tp>=need && fp<=cfg$false_allowance
  }
  list(tp=tp,fp=fp,sensitivity=sensitivity,false_positive_rate=fpr,
    specificity=if(is.null(fpr)) NULL else 1-fpr,success=success,
    positive_groups=positives,negative_groups=negatives)
}

# Studies are 1000 seeds apart (one study uses seed+0 to seed+205: covariates,
# outcomes, nuisance folds, multipliers), in blocks of 2500 (the replication
# limit) per grid size, so no two studies share a seed. app.py keeps the
# calibration seed outside these blocks.
study_seed <- function(cfg,n,replicate)
  as.integer(cfg$seed+1000L*(2500L*(match(n,cfg$sample_sizes)-1L)+replicate-1L))

orthogonal_tests <- function(d,nu,map,case) {
  # Stage 2's tests run in app.py's worker (orthogonal_tests.py), which reads
  # these values and writes modifier_tests.json in the study directory.
  port <- Sys.getenv('OCI_CALCULATOR_TESTS_PORT')
  if(!nzchar(port)) stop('The orthogonal_tests selector needs the worker that app.py starts.')
  inputs <- tempfile('study-'); dir.create(inputs); on.exit(unlink(inputs,recursive=TRUE))
  write.csv(map[c('coordinate','group','kind','contrast')],file.path(inputs,'columns.csv'),row.names=FALSE)
  writeBin(as.double(c(d$A,d$Y,nu$e,nu$mu0,nu$mu1,as.matrix(d[map$coordinate]))),
    file.path(inputs,'values.bin'),endian='little')
  con <- socketConnection('127.0.0.1',as.integer(port),blocking=TRUE,open='r+',timeout=86400)
  on.exit(close(con),add=TRUE)
  writeLines(paste(normalizePath(case),inputs,nrow(d),sep='\t'),con)
  reply <- readLines(con,n=1)
  if(!identical(reply,'ok')) stop('Orthogonal tests: ',if(length(reply)) reply else 'no reply from the worker')
  sort(as.character(unlist(fromJSON(file.path(case,'modifier_tests.json'))$selected)))
}

one_replication <- function(cfg,n,replicate,groups,cal,out) {
  seed <- study_seed(cfg,n,replicate)
  pop <- covariates(n,cfg,groups,seed)
  p <- probabilities(cfg,pop,cal)
  set.seed(seed+1L)
  A <- rbinom(n,1,p$e); Y <- rbinom(n,1,ifelse(A==1,p$mu1,p$mu0))
  d <- data.frame(pop$X,row_id=seq_len(n),A=A,Y=Y,p,check.names=FALSE)
  kept <- apply(pop$X,2,sd)>1e-12
  X <- pop$X[,kept,drop=FALSE]
  if(ncol(X)<2) stop('At least two nonconstant encoded candidates are needed by the reused glmnet routines.')
  case <- file.path(out,sprintf('n%d_r%03d',n,replicate)); dir.create(case)
  if(cfg$nuisance=='oracle') {
    nu <- derive_nuisances(d,p)
  } else {
    nu <- derive_nuisances(d,fit_nuisances(d,seq_len(n),seed+2L,file.path(case,'nuisance')))
  }
  if(identical(cfg$selector,'orthogonal_tests')) {
    selected <- orthogonal_tests(d,nu,pop$map,case); kkt <- NULL
  } else {
    set.seed(seed+3L); G <- matrix(rnorm(n*500),n,500)
    fit <- if(cfg$scale=='logOR') select_gao(X,Y,nu,G) else select_zhao(X,Y,nu,G)
    selected_coordinates <- colnames(X)[fit$selected]
    selected <- sort(unique(pop$map$group[pop$map$coordinate %in% selected_coordinates]))
    coefficient <- data.frame(coordinate=colnames(X),beta=fit$beta,center=fit$center,scale=fit$scale)
    write.csv(coefficient,file.path(case,'coefficients.csv'),row.names=FALSE)
    write_json_file(list(alpha=fit$alpha,lambda=fit$lambda,kkt=fit$kkt,selected=I(selected),
      dropped_columns=I(pop$map$coordinate[!kept])),file.path(case,'fit.json'))
    kkt <- fit$kkt
  }
  truth <- groups$group[abs(cal$weights$beta)>0]
  # Non-modifiers that shift the untreated risk modify the effect on the other scale.
  prognostic <- groups$group[abs(cal$weights$gamma)>0 & abs(cal$weights$beta)==0]
  detection <- detection_metrics(selected,truth,groups$group,cfg)
  tp <- detection$tp; fp <- detection$fp
  # First replication at each n is a complete independent-reconstruction fixture.
  if(replicate==1) {
    write.csv(d,file.path(case,'data.csv'),row.names=FALSE)
    write.csv(nu,file.path(case,'derived_nuisances.csv'),row.names=FALSE)
    write.csv(pop$map,file.path(case,'groups.csv'),row.names=FALSE)
  }
  list(n=n,replicate=replicate,seed=seed,selected=I(selected),truth=I(truth),tp=tp,fp=fp,
    exact=setequal(selected,truth),all_recovered=tp==length(truth),success=detection$success,
    sensitivity=detection$sensitivity,false_positive_rate=detection$false_positive_rate,specificity=detection$specificity,
    positive_groups=detection$positive_groups,negative_groups=detection$negative_groups,
    fp_prognostic=length(intersect(selected,prognostic)),
    error=NULL,kkt=kkt,event_rate=mean(Y),treated_rate=mean(A),
    expected_untreated_risk=mean(p$mu0),expected_treated_risk=mean(p$mu1),
    propensity_min=min(p$e),propensity_max=max(p$e),overlap_fraction=mean(p$e>=.1 & p$e<=.9),
    propensity_brier=mean((nu$e-p$e)^2),
    outcome_probability_mse=mean(c((nu$mu0-p$mu0)^2,(nu$mu1-p$mu1)^2)))
}

wilson <- function(k,n) {
  z <- qnorm(.975); phat <- k/n; denom <- 1+z*z/n
  mid <- (phat+z*z/(2*n))/denom
  half <- z*sqrt(phat*(1-phat)/n+z*z/(4*n*n))/denom
  c(lower=max(0,mid-half),upper=min(1,mid+half))
}

summarize_records <- function(records,cfg,groups,cal) {
  truth <- groups$group[abs(cal$weights$beta)>0]
  alpha <- .05/length(cfg$sample_sizes)
  cells <- lapply(cfg$sample_sizes,function(n) {
    rows <- Filter(function(x) x$n==n,records); R <- length(rows)
    k <- sum(vapply(rows,function(x) isTRUE(x$success),logical(1)))
    interval <- wilson(k,R)
    # Simultaneous one-sided exact lower bounds across the finite fixed grid.
    lower <- if(k==0) 0 else qbeta(alpha,k,R-k+1)
    avg <- function(key) mean(vapply(rows,function(x) if(is.null(x[[key]])) NA_real_ else as.numeric(x[[key]]),numeric(1)),na.rm=TRUE)
    frequency <- lapply(truth,function(g) {
      found <- sum(vapply(rows,function(x) g %in% x$selected,logical(1)))
      list(group=g,count=found,probability=found/R,interval=as.list(wilson(found,R)))
    })
    list(n=n,replications=R,successes=k,probability=k/R,interval=as.list(interval),lower_bound=lower,
      failures=sum(vapply(rows,function(x) !is.null(x$error),logical(1))),
      mean_true_selected=avg('tp'),mean_false_selected=avg('fp'),mean_false_selected_prognostic=avg('fp_prognostic'),
      mean_sensitivity=avg('sensitivity'),mean_specificity=avg('specificity'),mean_false_positive_rate=avg('false_positive_rate'),
      exact_probability=mean(vapply(rows,function(x) isTRUE(x$exact),logical(1))),
      all_recovered_probability=mean(vapply(rows,function(x) isTRUE(x$all_recovered),logical(1))),
      any_selected_probability=mean(vapply(rows,function(x) length(x$selected)>0,logical(1))),
      overlap_fraction=avg('overlap_fraction'),event_rate=avg('event_rate'),treated_rate=avg('treated_rate'),
      propensity_brier=avg('propensity_brier'),outcome_probability_mse=avg('outcome_probability_mse'),
      modifier_frequencies=frequency)
  })
  meets <- vapply(cells,function(x) x$lower_bound>=cfg$target_probability && x$failures==0,logical(1))
  eligible <- which(vapply(seq_along(meets),function(i) all(meets[i:length(meets)]),logical(1)))
  candidate <- if(cfg$mode=='pilot' || length(eligible)==0) NULL else cfg$sample_sizes[min(eligible)]
  output_config <- cfg; output_config$sample_sizes <- I(cfg$sample_sizes)
  list(config=output_config,truth=I(truth),method=if(identical(cfg$selector,'orthogonal_tests')) 'Stage 2 orthogonal tests (Cauchy combination, BH)' else
      if(cfg$scale=='logOR') 'Gao DINA + sparse selection' else 'Zhao residual lasso',
    candidate_n=candidate,recommended_n=NULL,
    status=if(cfg$mode=='pilot') 'Pilot only — no sample-size recommendation' else if(is.null(candidate))
      'Target not established on the evaluated grid' else 'Provisional grid crossing — independent confirmation required',
    uncertainty='95% Wilson intervals; grid decision uses simultaneous one-sided 95% exact binomial lower bounds.',
    cells=cells)
}
