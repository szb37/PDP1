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

preds = c('gender',	'edu',	'age',	'LED',	'severity',	'is_anx',	'is_dep',	
          'fivedasc_util_total',	'fivedasc_sprit_total',	'fivedasc_bliss_total',	
          'fivedasc_insight_total',	'fivedasc_dis_total',	'fivedasc_imp_total',	
          'fivedasc_anx_total',	'fivedasc_cimg_total',	'fivedasc_eimg_total',	
          'fivedasc_av_total',	'fivedasc_per_total',	
          'boundlessMean',	'anxiousEgoMean',	'visionaryMean')


### Do models
df_stats <- data.frame()
for (this_measure in unique(df_data$measure)) {
  for (this_pred in preds) {
    
    ### do mixed model
    model_spec=paste('score~(1|pID)+tp*',this_pred,sep='')
    model <- lmer(model_spec, subset(df_data, measure==this_measure))
    coeffs <- rownames_to_column(data.frame(coef(summary(model))), var = 'tp')
    coeffs$measure <- this_measure
    df_stats <- rbind(df_stats, coeffs)
    
  }
}

### Format df_data
df_stats$est <- round(df_stats$Estimate, 2)
df_stats$Estimate <- NULL
df_stats$SE <- round(df_stats$Std..Error, 2)
df_stats$Std..Error <- NULL
df_stats$p.value <- df_stats$Pr...t..
df_stats$Pr...t.. <- NULL
df_stats$df <- round(df_stats$df, 1)
df_stats$t.value <- round(df_stats$t.value, 2)

### Add significances
df_stats <- df_stats %>%
  mutate(sig =
    ifelse(p.value<=0.001, '***',
    ifelse(p.value<=0.01, '**',
    ifelse(p.value<=0.05, '*', ''))))

df_stats <- df_stats %>%
  mutate(p.value = ifelse(
    p.value<=0.001, 
    '<0.001', 
    format(round(df_stats$p.value,3), nsmall=3)))

### Format and save
df_stats$tp <- sub('(Intercept)', 'Intercept', df_stats$tp)
df_stats$tp <- sub('tpA1',  'tp(A1)', df_stats$tp)
df_stats$tp <- sub('tpA7',  'tp(A7)', df_stats$tp)
df_stats$tp <- sub('tpB1',  'tp(B1)', df_stats$tp)
df_stats$tp <- sub('tpB7',  'tp(B7)', df_stats$tp)
df_stats$tp <- sub('tpB30', 'tp(B30)', df_stats$tp)
df_stats$tp <- sub('tpB90', 'tp(B90)', df_stats$tp)

df_stats <- df_stats[, c("measure", "tp", "est", "SE", "df", "t.value", "p.value", "sig")]  
export_dir <- paste(here(),'/exports',sep='')
write.csv(df_stats, file=paste(export_dir,'/pdp1_predictors_mixedmod.csv', sep=''), row.names=FALSE)