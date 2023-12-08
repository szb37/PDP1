rm(list = ls())
library(performance)
library(lmerTest)
library(tibble)
library(tidyr)
library(dplyr)
library(here)

source(paste(here(),'/R models/pdp1_helpers.r', sep=''))
df_data <- load_pdp1_data()

### Do models
df <- data.frame()
for (this_measure in unique(df_data$measure)) {

  model <- lmer('score~(1|pID)+tp', subset(df_data, measure==this_measure))
  coeffs <- rownames_to_column(data.frame(coef(summary(model))), var = 'tp')
  coeffs$measure <- this_measure
  coeffs$adj.p.value <- NA
  coeffs$hedges.g <- NA
  df <- rbind(df, coeffs)
}

### Format df_data
df <- format_mixedmodel_output(df)

### Add Bonferroni adjusted p-values for cognitive tests
for (task in unique(df_data$test)){
  for (tp in unique(df$tp)){

    #   For tests wo multiple outcomes adj-p-values should be same as unadjusted
    #   Adjustment should be ignored for UPDRS as here different test components
    #   are used as indep measures in the literature        
    subdf <- df[grepl(task, df$measure) & grepl(tp, df$tp), ]
    
    if(nrow(subdf)==0){
      next
    }
    
    subdf$adj.p.value <- p.adjust(subdf$p.value, method='bonferroni')
    df[grepl(task, df$measure) & grepl(tp, df$tp), ] <- subdf

  }
}

### Add sig stars

### Add significances
df <- df %>%
  mutate(sig =
           ifelse(p.value<=0.001, '***',
                  ifelse(p.value<=0.01, '**',
                         ifelse(p.value<=0.05, '*', ''))))

df <- df %>%
  mutate(p.value = ifelse(
    p.value<=0.001, 
    '<0.001', 
    format(round(df$p.value,3), nsmall=3)))


df <- df %>%
  mutate(adj.sig =
           ifelse(adj.p.value<=0.001, '***',
                  ifelse(adj.p.value<=0.01, '**',
                         ifelse(adj.p.value<=0.05, '*', ''))))

df <- df %>%
  mutate(adj.p.value = ifelse(
    adj.p.value<=0.001, 
    '<0.001', 
    format(round(df$adj.p.value, 3), nsmall=3)))




### Save results
#df <- rename_timepoints(df)
#df <- df[, c("measure", "tp", "est", "SE", "hedges.g", "df", "t.value", "p.value", "sig", "adj.p.value", "adj.sig")]  
#export_dir <- paste(here(),'/exports',sep='')
#write.csv(df, file=paste(export_dir,'/tmp_pdp1_mixed_models_v1.csv', sep=''), row.names=FALSE)