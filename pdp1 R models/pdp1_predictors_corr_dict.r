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
dict_preds = c('gender',	'is_anx',	'is_dep')


for (this_measure in unique(df_data$measure)) {
  for (this_tp in unique(df_data$tp)) {
    for (this_pred in dict_preds) {
      
      scores = subset(df_data, (measure==this_measure) & (tp==this_tp))[, 'score']
      pred_vals = subset(df_data, (measure==this_measure) & (tp==this_tp))[, this_pred]
      
      biserial <- cor.test(scores, as.numeric(pred_vals))
      
      tmp <- data.frame(
        'measure'= this_measure, 
        'tp'= this_tp, 
        'pred'= this_pred,
        'biserial.p'= biserial$p.value
      )
      
      df_stats <- rbind(df_stats, tmp)
      
    }
  }
}

df_stats <- df_stats %>%
  mutate(biserial.sig =
    ifelse(biserial.p<=0.001, '***',
    ifelse(biserial.p<=0.01, '**',
    ifelse(biserial.p<=0.05, '*', ''))))

df_stats$biserial.p <- round(df_stats$biserial.p, 3)

export_dir <- paste(here(),'/exports',sep='')
write.csv(df_stats, file=paste(export_dir,'/pdp1_predictors_corr_dict.csv', sep=''), row.names=FALSE)