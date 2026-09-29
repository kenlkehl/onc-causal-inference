# Gao and Zhao sparse selectors used by the standalone Note 030 experiment.
# Source-only module. Functions perform no work until explicitly called.
suppressPackageStartupMessages({library(glmnet); library(jsonlite)})
write_json_file <- function(x,path) write_json(x,path,auto_unbox=TRUE,pretty=TRUE,digits=17,na='null',null='null')
write_csv <- function(x,path) write.csv(x,path,row.names=FALSE)
clip_probability <- function(p) {
  # In-place replacement preserves matrix dimensions (pmin(scalar, matrix) does not).
  p[p<.01] <- .01
  p[p>.99] <- .99
  p
}

stratified_folds <- function(a,k,seed) {
  set.seed(seed)
  answer <- integer(length(a))
  for(value in sort(unique(a))) {
    ix <- which(a==value)
    answer[ix] <- sample(rep(seq_len(k),length.out=length(ix)))
  }
  answer
}

fit_nuisances <- function(d,fit_rows,seed,save_dir) {
  # Every fitted row belongs to fit_rows; other rows are prediction-only.
  dir.create(save_dir,recursive=TRUE)
  X <- as.matrix(d[grepl('^x[0-9]+$',names(d))])
  folds <- stratified_folds(d$A[fit_rows],2,seed)
  predictions <- array(NA_real_,c(nrow(d),3,2))
  support <- matrix(FALSE,ncol(X),3)
  lineage <- list(); coefficient_rows <- list()
  for(fold in 1:2) {
    source_rows <- fit_rows[folds!=fold]
    for(task in 1:3) {
      rows <- if(task==1) source_rows else source_rows[d$A[source_rows]==task-2]
      response <- if(task==1) d$A[rows] else d$Y[rows]
      cvfolds <- stratified_folds(response,5,seed+100*fold+task)
      model <- cv.glmnet(X[rows,,drop=FALSE],response,family='binomial',alpha=1,
        foldid=cvfolds,nlambda=50,type.measure='deviance',standardize=TRUE,
        intercept=TRUE,thresh=1e-12,maxit=100000)
      predictions[,task,fold] <- as.numeric(predict(model,X,s='lambda.min',type='response'))
      co <- as.numeric(coef(model,s='lambda.min'))
      support[,task] <- support[,task] | abs(co[-1])>1e-7
      coefficient_rows[[length(coefficient_rows)+1]] <- data.frame(fold=fold,task=task,
        feature=c('(intercept)',colnames(X)),coefficient=co,lambda=model$lambda.min)
      lineage[[length(lineage)+1]] <- list(fold=fold,task=task,
        training_row_ids=d$row_id[rows],cv_fold=cvfolds,lambda=model$lambda.min)
    }
  }
  p <- apply(predictions,c(1,2),mean)
  for(fold in 1:2) {
    rows <- fit_rows[folds==fold]
    p[rows,] <- predictions[rows,,fold]
  }
  clipped <- p<.01 | p>.99
  p <- clip_probability(p)
  result <- data.frame(row_id=d$row_id,e=p[,1],mu0=p[,2],mu1=p[,3])
  write_csv(result,file.path(save_dir,'predictions.csv'))
  write_csv(data.frame(feature=colnames(X),e=support[,1],mu0=support[,2],mu1=support[,3],
    selected=apply(support,1,any)),file.path(save_dir,'screening.csv'))
  write_csv(do.call(rbind,coefficient_rows),file.path(save_dir,'coefficients.csv'))
  write_json_file(list(fit_row_ids=d$row_id[fit_rows],fold_id=folds,models=lineage,
    clipped_count=colSums(clipped)),file.path(save_dir,'lineage.json'))
  result
}

derive_nuisances <- function(d,p) {
  V0 <- p$mu0*(1-p$mu0); V1 <- p$mu1*(1-p$mu1)
  a <- p$e*V1/(p$e*V1+(1-p$e)*V0)
  nu <- a*qlogis(p$mu1)+(1-a)*qlogis(p$mu0)
  m <- (1-p$e)*p$mu0+p$e*p$mu1
  data.frame(row_id=d$row_id,e=p$e,mu0=p$mu0,mu1=p$mu1,m=m,a=a,nu=nu,
    u_rd=d$A-p$e,r=d$Y-m,u_dina=d$A-a,
    variance=ifelse(d$A==1,V1,V0))
}

make_design <- function(X,u) {
  center <- colMeans(X)
  D0 <- sweep(X,2,center,'-')*u
  scale <- apply(D0,2,sd)
  stopifnot(all(scale>1e-12))
  list(center=center,scale=scale,D=sweep(D0,2,scale,'/'))
}

select_zhao <- function(X,Y,nu,G) {
  u <- nu$u_rd; r <- nu$r
  ds <- make_design(X,u); D <- ds$D
  W <- D-u%*%(crossprod(u,D)/sum(u^2))
  yp <- as.numeric(r-u*sum(u*r)/sum(u^2))
  full <- lm.fit(cbind(u,D),r)
  stopifnot(full$rank==ncol(D)+1)
  sigma <- sqrt(sum(full$residuals^2)/(nrow(X)-full$rank))
  scores <- apply(abs(crossprod(W,G*sigma)),2,max)
  lambda <- 1.1*mean(scores)
  fit <- glmnet(W,yp,lambda=lambda/nrow(X),alpha=1,intercept=FALSE,
    standardize=FALSE,thresh=1e-12,maxit=100000)
  beta <- as.numeric(fit$beta[,1]); selected <- abs(beta)>1e-7
  alpha <- sum(u*(r-as.numeric(D%*%beta)))/sum(u^2)
  score <- as.numeric(crossprod(D,r-u*alpha-as.numeric(D%*%beta)))
  nz <- abs(beta)>1e-10
  kkt <- max(c(abs(score[nz]-lambda*sign(beta[nz])),pmax(abs(score[!nz])-lambda,0),
    abs(sum(u*(r-u*alpha-as.numeric(D%*%beta))))),0)/nrow(X)
  stopifnot(is.finite(kkt),kkt<1e-5)
  list(selected=which(selected),beta=beta,alpha=alpha,center=ds$center,scale=ds$scale,
    lambda=lambda,sigma=sigma,kkt=kkt,scores=scores,objective='Zhao Eq7 squared residual loss')
}

select_gao <- function(X,Y,nu,G) {
  u <- nu$u_dina; v <- nu$variance
  ds <- make_design(X,u); D <- ds$D
  W <- D-u%*%(crossprod(u*v,D)/sum(v*u^2))
  scores <- apply(abs(crossprod(W,G*sqrt(v))),2,max)
  lambda <- 1.1*mean(scores)
  p <- ncol(X)
  # glmnet rescales penalty.factor to sum to p+1, hence p/(p+1).
  fit <- glmnet(cbind(u,D),Y,offset=nu$nu,family='binomial',alpha=1,
    lambda=lambda/nrow(X)*p/(p+1),penalty.factor=c(0,rep(1,p)),
    intercept=FALSE,standardize=FALSE,thresh=1e-12,maxit=100000)
  co <- as.numeric(fit$beta[,1]); alpha <- co[1]; beta <- co[-1]
  prediction <- plogis(nu$nu+u*alpha+as.numeric(D%*%beta))
  score <- as.numeric(crossprod(D,Y-prediction)); nz <- abs(beta)>1e-10
  kkt <- max(c(abs(score[nz]-lambda*sign(beta[nz])),pmax(abs(score[!nz])-lambda,0),
    abs(sum(u*(Y-prediction)))),0)/nrow(X)
  stopifnot(is.finite(kkt),kkt<1e-5)
  list(selected=which(abs(beta)>1e-7),beta=beta,alpha=alpha,center=ds$center,scale=ds$scale,
    lambda=lambda,sigma=NA_real_,kkt=kkt,scores=scores,objective='Gao DINA binomial + proposed L1 selection')
}
