rm(list = ls())
library(performance)
library(lmerTest)
library(tibble)
library(tidyr)
library(dplyr)
library(here)

filepath = paste(here(),'/data/pdp1_master_covs.csv', sep ='')
df_data <- read.csv(filepath, header=TRUE, sep = ",", stringsAsFactors=FALSE)
df_data$tp <- factor(df_data$tp, levels=c('bsl', 'A1', 'A7', 'B1', 'B7', 'B30', 'B90'))
df_data$edu <- as.factor(df_data$edu)
df_data$gender <- as.factor(df_data$gender)
df_data$is_anx <- as.logical(df_data$is_anx)
df_data$is_dep <- as.logical(df_data$is_dep)

df_stats <- data.frame()
cont_preds = c('age', 'LED',	'severity',
          'fivedasc_util_total',	'fivedasc_sprit_total',	'fivedasc_bliss_total',	
          'fivedasc_insight_total',	'fivedasc_dis_total',	'fivedasc_imp_total',	
          'fivedasc_anx_total',	'fivedasc_cimg_total',	'fivedasc_eimg_total',	
          'fivedasc_av_total',	'fivedasc_per_total',	
          'boundlessMean',	'anxiousEgoMean',	'visionaryMean')

for (this_measure in unique(df_data$measure)) {
  for (this_tp in unique(df_data$tp)) {
    for (this_pred in cont_preds) {
      
      scores = subset(df_data, (measure==this_measure) & (tp==this_tp))[, 'score']
      pred_vals = subset(df_data, (measure==this_measure) & (tp==this_tp))[, this_pred]
      
      pearson <- cor.test(scores, pred_vals, method = "pearson")
      spearman <- cor.test(scores, pred_vals, method = "spearman")
      kendall <- cor.test(scores, pred_vals, method = "kendall")
      
      tmp <- data.frame(
          'measure'= this_measure, 
          'tp'= this_tp, 
          'pred'= this_pred,
          'pearson.p'= pearson$p.value,
          'spearman.p'= spearman$p.value,
          'kendall.p'= kendall$p.value
          )
      
      df_stats <- rbind(df_stats, tmp)
    
    }
  }
}

df_stats <- df_stats %>%
  mutate(pearson.sig =
    ifelse(pearson.p<=0.001, '***',
    ifelse(pearson.p<=0.01, '**',
    ifelse(pearson.p<=0.05, '*', ''))))

df_stats <- df_stats %>%
  mutate(spearman.sig =
    ifelse(spearman.p<=0.001, '***',
    ifelse(spearman.p<=0.01, '**',
    ifelse(spearman.p<=0.05, '*', ''))))

df_stats <- df_stats %>%
  mutate(kendall.sig =
    ifelse(kendall.p<=0.001, '***',
    ifelse(kendall.p<=0.01, '**',
    ifelse(kendall.p<=0.05, '*', ''))))

df_stats$pearson.p <- round(df_stats$pearson.p, 3)
df_stats$spearman.p <- round(df_stats$spearman.p, 3)
df_stats$kendall.p <- round(df_stats$kendall.p, 3)

export_dir <- paste(here(),'/exports',sep='')
write.csv(df_stats, file=paste(export_dir,'/pdp1_predictors_corr_cont.csv', sep=''), row.names=FALSE)