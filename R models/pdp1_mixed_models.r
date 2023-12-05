rm(list = ls())
library(performance)
library(lmerTest)
library(tibble)
library(tidyr)
library(dplyr)
library(here)

source(paste(here(),'/R models/load_pdp1.r', sep=''))
df_data <- load_pdp1()

### Do models
df_stats <- data.frame()
for (this_measure in unique(df_data$measure)) {

  model <- lmer('score~(1|pID)+tp', subset(df_data, measure==this_measure))
  coeffs <- rownames_to_column(data.frame(coef(summary(model))), var = 'tp')
  coeffs$measure <- this_measure
  coeffs$adj.p.value <- NA
  df_stats <- rbind(df_stats, coeffs)
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

### Add Bonferoni adjusted p-values for cognitive tests
#   For tests wo multiple outcomes adj-p-values should be same as unadjusted
#   Adjustment should be ignored for UPDRS as here different test components
#   are used as indep measures in the literature

for (task in unique(df_data$test)){
  for (tp in unique(df_stats$tp)){
        
    subdf <- df_stats[grepl(task, df_stats$measure) & grepl(tp, df_stats$tp), ]
    
    if(nrow(subdf)==0){
      next
    }
    
    subdf$adj.p.value <- p.adjust(subdf$p.value, method='bonferroni')
    df_stats[grepl(task, df_stats$measure) & grepl(tp, df_stats$tp), ] <- subdf

  }
}

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


df_stats <- df_stats %>%
  mutate(adj.sig =
    ifelse(adj.p.value<=0.001, '***',
    ifelse(adj.p.value<=0.01, '**',
    ifelse(adj.p.value<=0.05, '*', ''))))

df_stats <- df_stats %>%
  mutate(adj.p.value = ifelse(
    adj.p.value<=0.001, 
    '<0.001', 
    format(round(df_stats$adj.p.value, 3), nsmall=3)))


### Format and save
df_stats$tp <- sub('(Intercept)', 'Intercept', df_stats$tp)
df_stats$tp <- sub('tpA1',  'tp(A1)', df_stats$tp)
df_stats$tp <- sub('tpA7',  'tp(A7)', df_stats$tp)
df_stats$tp <- sub('tpB1',  'tp(B1)', df_stats$tp)
df_stats$tp <- sub('tpB7',  'tp(B7)', df_stats$tp)
df_stats$tp <- sub('tpB30', 'tp(B30)', df_stats$tp)
df_stats$tp <- sub('tpB90', 'tp(B90)', df_stats$tp)




df_stats <- df_stats[, c("measure", "tp", "est", "SE", "df", "t.value", "p.value", "sig", "adj.p.value", "adj.sig")]  
export_dir <- paste(here(),'/exports',sep='')
write.csv(df_stats, file=paste(export_dir,'/pdp1_mixed_models_v1.csv', sep=''), row.names=FALSE)