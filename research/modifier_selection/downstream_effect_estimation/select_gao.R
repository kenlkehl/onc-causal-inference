#!/usr/bin/env Rscript
suppressPackageStartupMessages({library(glmnet); library(jsonlite)})

args <- commandArgs(trailingOnly=TRUE)
if(length(args) != 4) stop('usage: select_gao.R INPUT_CSV GROUPS_CSV OUTPUT_JSON METHODS_R')
source(args[[4]])

d <- read.csv(args[[1]], check.names=FALSE)
groups <- read.csv(args[[2]], check.names=FALSE)
feature_names <- groups$coordinate_name
X <- as.matrix(d[feature_names])
p <- data.frame(row_id=d$row_id, e=d$e, mu0=d$mu0, mu1=d$mu1)
nu <- derive_nuisances(d, p)

# The multiplier process and seed match the sparse Note 030 implementation.
set.seed(unique(d$selection_seed))
G <- matrix(rnorm(nrow(d) * 500), nrow(d), 500)
fit <- select_gao(X, d$Y, nu, G)
selected_coordinates <- fit$selected - 1L
selected_groups <- sort(unique(groups$candidate_name[fit$selected]))

write_json_file(list(
  selected_coordinates=selected_coordinates,
  selected_groups=selected_groups,
  coefficients=fit$beta,
  alpha=fit$alpha,
  lambda=fit$lambda,
  kkt=fit$kkt
), args[[3]])
