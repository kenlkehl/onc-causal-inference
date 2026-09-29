# Fit Gao and Zhao selectors for the standalone Note 030 reproduction.
# This file is called by run.py; use run.py as the public entry point.

args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==2)
out <- normalizePath(args[1],mustWork=TRUE)
methods_file <- normalizePath(args[2],mustWork=TRUE)
source(methods_file)
RNGkind('Mersenne-Twister','Inversion','Rejection')

prepared <- read.csv(file.path(out,'prepared_base_tables.csv'),stringsAsFactors=FALSE)

balanced_groups <- function() {
  data.frame(
    coordinate=0:109,
    coordinate_name=sprintf('x%03d',0:109),
    candidate_index=0:109,
    candidate_name=c(paste0('C',1:5),paste0('M',1:5),paste0('N',sprintf('%03d',1:100))),
    display_name=c(paste0('C',1:5),paste0('M',1:5),paste0('N',sprintf('%03d',1:100))),
    role=c(rep('confounder',5),rep('modifier',5),rep('noise',100))
  )
}

# Brute-force balanced condition: exact Note 030 R generator.
for(ii in 1:10) {
  source_seed <- 740000L+ii
  stem <- sprintf('balanced__n12800__r%02d',ii)
  set.seed(source_seed)
  X <- matrix(sample(c(-1,1),12800*110,replace=TRUE),12800,110)
  colnames(X) <- sprintf('x%03d',0:109)
  C <- X[,1:5,drop=FALSE]
  M <- X[,6:10,drop=FALSE]
  e <- plogis(.8*rowSums(C)/sqrt(5))
  mu0 <- plogis(-.4+.5*rowSums(C)/sqrt(5))
  truth_logor <- .3+as.numeric(M%*%(.6*c(1,-1,1,-1,1)))
  mu1 <- plogis(qlogis(mu0)+truth_logor)
  A <- rbinom(12800,1,e)
  Y <- rbinom(12800,1,ifelse(A==1,mu1,mu0))
  G <- matrix(rnorm(12800*500),12800,500)
  inner_fold <- stratified_folds(A,2,source_seed+9000)-1L
  d <- data.frame(
    X,row_id=1:12800,A=A,Y=Y,e=e,mu0=mu0,mu1=mu1,
    truth_rd=mu1-mu0,truth_logor=truth_logor,inner_fold=inner_fold,
    check.names=FALSE
  )
  write.csv(d,file.path(out,'inputs',paste0(stem,'.csv')),row.names=FALSE)
  write.csv(balanced_groups(),file.path(out,'groups',paste0(stem,'.csv')),row.names=FALSE)
  write.table(
    G,gzfile(file.path(out,'gaussian',paste0(stem,'.csv.gz'))),
    sep=',',row.names=FALSE,col.names=FALSE
  )
  prepared <- rbind(prepared,data.frame(
    stem=stem,mechanism='balanced',n=12800,replicate=ii,source_seed=source_seed
  ))
}

# NSCLC expansions preserve the empirical 800-row clinical distribution while
# redrawing treatment, outcome, noise covariates, and multiplier variables.
for(ii in 1:10) {
  source_seed <- 620000L+ii
  source_stem <- sprintf('nsclc_source__r%02d',ii)
  d0 <- read.csv(file.path(out,'sources',paste0(source_stem,'.csv')))
  groups <- read.csv(file.path(out,'sources',paste0(source_stem,'__groups.csv')))
  for(n in c(3200L,12800L)) {
    multiplier <- n %/% 800L
    expansion_seed <- if(n==3200L) 730000L+ii else 750000L+ii
    stem <- sprintf('nsclc__n%d__r%02d',n,ii)
    set.seed(expansion_seed)
    index <- sample(rep(1:800,multiplier))
    clinical <- as.matrix(d0[index,sprintf('x%03d',0:17)])
    noise <- matrix(rnorm(n*100),n,100)
    X <- cbind(clinical,noise)
    colnames(X) <- sprintf('x%03d',0:117)
    e <- d0$e[index]
    mu0 <- d0$mu0[index]
    mu1 <- d0$mu1[index]
    A <- rbinom(n,1,e)
    Y <- rbinom(n,1,ifelse(A==1,mu1,mu0))
    G <- matrix(rnorm(n*500),n,500)
    inner_fold <- stratified_folds(A,2,expansion_seed+9000)-1L
    d <- data.frame(
      X,row_id=1:n,A=A,Y=Y,e=e,mu0=mu0,mu1=mu1,
      truth_rd=mu1-mu0,truth_logor=qlogis(mu1)-qlogis(mu0),
      inner_fold=inner_fold,check.names=FALSE
    )
    write.csv(d,file.path(out,'inputs',paste0(stem,'.csv')),row.names=FALSE)
    write.csv(groups,file.path(out,'groups',paste0(stem,'.csv')),row.names=FALSE)
    write.table(
      G,gzfile(file.path(out,'gaussian',paste0(stem,'.csv.gz'))),
      sep=',',row.names=FALSE,col.names=FALSE
    )
    prepared <- rbind(prepared,data.frame(
      stem=stem,mechanism='nsclc',n=n,replicate=ii,source_seed=source_seed
    ))
  }
}

prepared <- prepared[order(prepared$mechanism,prepared$n,prepared$replicate),]
write.csv(prepared,file.path(out,'prepared_tables.csv'),row.names=FALSE)

result_rows <- list()
coefficient_rows <- list()
warning_rows <- list()

count_fit <- function(fit,groups) {
  selected_coordinates <- fit$selected-1L
  selected_groups <- unique(groups$candidate_index[groups$coordinate %in% selected_coordinates])
  group_roles <- unique(groups[c('candidate_index','role')])
  roles <- group_roles$role[group_roles$candidate_index %in% selected_groups]
  modifiers <- sort(unique(groups$candidate_name[
    groups$coordinate %in% selected_coordinates & groups$role=='modifier'
  ]))
  list(
    selected_m=sum(roles=='modifier'),
    selected_n=sum(roles=='noise'),
    modifiers=paste(modifiers,collapse=';')
  )
}

for(i in seq_len(nrow(prepared))) {
  spec <- prepared[i,]
  d <- read.csv(file.path(out,'inputs',paste0(spec$stem,'.csv')))
  X <- as.matrix(d[grepl('^x[0-9]+$',names(d))])
  rows <- seq_len(nrow(d))
  groups <- read.csv(file.path(out,'groups',paste0(spec$stem,'.csv')),check.names=FALSE)
  G <- as.matrix(read.csv(
    gzfile(file.path(out,'gaussian',paste0(spec$stem,'.csv.gz'))),header=FALSE
  ))
  for(regime in c('oracle','estimated')) {
    task <- function() {
      if(regime=='oracle') {
        nuisance <- derive_nuisances(d,data.frame(e=d$e,mu0=d$mu0,mu1=d$mu1))
      } else {
        common <- fit_nuisances(
          d,rows,spec$source_seed+1000,
          file.path(out,'tables',spec$stem,'estimated_common')
        )
        nuisance <- derive_nuisances(d,common)
        for(fold in 0:1) {
          source_rows <- rows[d$inner_fold!=fold]
          fit_nuisances(
            d,source_rows,spec$source_seed+2000+fold,
            file.path(out,'tables',spec$stem,paste0('estimated_core_fold',fold))
          )
        }
      }
      for(method in c('zhao','gao_sparse')) {
        fit <- if(method=='zhao') {
          select_zhao(X,d$Y,nuisance,G)
        } else {
          select_gao(X,d$Y,nuisance,G)
        }
        counts <- count_fit(fit,groups)
        result_rows[[length(result_rows)+1]] <<- data.frame(
          mechanism=spec$mechanism,n=spec$n,nuisance=regime,
          replicate=spec$replicate,source_seed=spec$source_seed,method=method,
          selected_m=counts$selected_m,selected_n=counts$selected_n,
          modifiers=counts$modifiers
        )
        coefficient_rows[[length(coefficient_rows)+1]] <<- data.frame(
          stem=spec$stem,nuisance=regime,method=method,
          coordinate=0:(ncol(X)-1),beta=fit$beta,
          selected=abs(fit$beta)>1e-7
        )
      }
    }
    withCallingHandlers(task(),warning=function(w) {
      warning_rows[[length(warning_rows)+1]] <<- data.frame(
        stem=spec$stem,nuisance=regime,message=conditionMessage(w)
      )
      invokeRestart('muffleWarning')
    })
  }
  cat('fit',spec$stem,'\n')
  flush.console()
}

write.csv(do.call(rbind,result_rows),file.path(out,'sparse_replications.csv'),row.names=FALSE)
write.csv(do.call(rbind,coefficient_rows),file.path(out,'sparse_coefficients.csv'),row.names=FALSE)
write_json_file(
  if(length(warning_rows)) do.call(rbind,warning_rows) else list(),
  file.path(out,'fit_warnings.json')
)
write_json_file(
  list(
    R=R.version.string,
    glmnet=as.character(packageVersion('glmnet')),
    jsonlite=as.character(packageVersion('jsonlite'))
  ),
  file.path(out,'R_runtime.json')
)
