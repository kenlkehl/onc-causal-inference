args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
cfg_path <- normalizePath(args[1]); out <- normalizePath(args[2]); methods <- normalizePath(args[3])
source(methods)
source(file.path(dirname(methods),'engine.R'))
cfg <- fromJSON(cfg_path,simplifyVector=TRUE)
atomic_json <- function(x,path) { write_json_file(x,paste0(path,'.tmp')); file.rename(paste0(path,'.tmp'),path) }
# Cooperative controls are checked before/after each complete study. A fit is
# never interrupted midway, so checkpoints contain only complete records.
checkpoint <- function(done,total) {
  repeat {
    control <- fromJSON(file.path(out,'control.json'))$action
    if(control=='cancelled') {
      atomic_json(list(state='cancelled',done=done,total=total,
        message='Cancelled at a study boundary; completed records retained; no sample-size recommendation.'),file.path(out,'progress.json'))
      return(FALSE)
    }
    if(control!='paused') {
      atomic_json(list(state='running',done=done,total=total),file.path(out,'progress.json'))
      return(TRUE)
    }
    atomic_json(list(state='paused',done=done,total=total),file.path(out,'progress.json'))
    Sys.sleep(.25)
  }
}
main <- function() {
  total <- length(cfg$sample_sizes)*cfg$replications
  if(!checkpoint(0L,total)) return(invisible(NULL))
  groups <- make_groups(cfg); cal <- calibrate(cfg,groups)
  write.csv(groups,file.path(out,'groups.csv'),row.names=FALSE)
  write_json_file(cal,file.path(out,'calibration.json'))
  write_json_file(list(R=R.version.string,glmnet=as.character(packageVersion('glmnet')),
    jsonlite=as.character(packageVersion('jsonlite'))),file.path(out,'R_runtime.json'))
  records <- list(); done <- 0L; total <- length(cfg$sample_sizes)*cfg$replications
  for(n in cfg$sample_sizes) for(rep in seq_len(cfg$replications)) {
    if(!checkpoint(done,total)) return(invisible(NULL))
    warnings <- character()
    record <- tryCatch(withCallingHandlers(one_replication(cfg,n,rep,groups,cal,out),warning=function(w) {
      warnings <<- c(warnings,conditionMessage(w)); invokeRestart('muffleWarning')
    }),error=function(e) list(n=n,replicate=rep,seed=study_seed(cfg,n,rep),
      selected=I(character()),truth=I(groups$group[abs(cal$weights$beta)>0]),tp=NULL,fp=NULL,
      success=FALSE,exact=FALSE,all_recovered=FALSE,error=conditionMessage(e)))
    record$warnings <- unique(warnings); records[[length(records)+1L]] <- record; done <- done+1L
    atomic_json(records,file.path(out,'replications.json'))
    atomic_json(list(state='running',done=done,total=total,n=n,replicate=rep),file.path(out,'progress.json'))
    if(!checkpoint(done,total)) return(invisible(NULL))
    cat(sprintf('%d/%d n=%d rep=%d error=%s\n',done,total,n,rep,if(is.null(record$error)) 'none' else record$error)); flush.console()
  }
  result <- summarize_records(records,cfg,groups,cal)
  atomic_json(result,file.path(out,'result.json'))
  atomic_json(list(state='complete',done=done,total=total),file.path(out,'progress.json'))
}
tryCatch(main(),error=function(e) {
  atomic_json(list(state='failed',error=conditionMessage(e)),file.path(out,'progress.json')); quit(status=1)
})
