rm(list = ls())
library(performance)
library(lmerTest)
library(tibble)
library(tidyr)
library(dplyr)
library(here)

source(paste(here(),'/pdp1 R models/pdp1_helpers.r', sep=''))
df_data <- load_pdp1_data()

### Do models
df_stats <- data.frame()
for (this_measure in unique(df_data$measure)) {

  print(this_measure)
  model <- lmer('score~(1|pID)+tp', subset(df_data, measure==this_measure))
  coeffs <- rownames_to_column(data.frame(coef(summary(model))), var = 'tp')
  coeffs$measure <- this_measure
  coeffs$adj.p.value <- NA
  
  # Add Hedges' g if needed
  if (TRUE){
    coeffs$hedges.g <- ''
    for (this_tp in c('A7', 'B7', 'B30')) {
      g = calc_hedges.g(
        subset(df_data, (measure==this_measure) & (tp==this_tp))$score,
        subset(df_data, (measure==this_measure) & (tp=='bsl'))$score)
      
      row_index <- which(coeffs$tp==paste('tp',this_tp,sep=''))
      coeffs$hedges.g[row_index[1]] <- g
    }
  }

  df_stats <- rbind(df_stats, coeffs)
}

df_stats <- format_mixedmodel_output(df_stats)

### Add Bonferoni adjusted p-values if needed
if (TRUE){
  for (task in unique(df_data$test)){
    for (tp in unique(df_stats$tp)){
      
      #   For tests wo multiple outcomes adj-p-values should be same as unadjusted
      #   Adjustment should be ignored for UPDRS as here different test components
      #   are used as indep measures in the literature        
      subdf <- df_stats[grepl(task, df_stats$measure) & grepl(tp, df_stats$tp), ]
      
      if(nrow(subdf)==0){
        next
      }
      subdf$adj.p.value <- p.adjust(subdf$p.value, method='bonferroni')
      df_stats[grepl(task, df_stats$measure) & grepl(tp, df_stats$tp), ] <- subdf
      
    }
  }
}

### Add sig stars if needed 
if (TRUE){
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
}

### Save
df_stats <- rename_timepoints(df_stats)
df_stats <- df_stats[, c("measure", "tp", "est", "SE", "hedges.g", "df", "t.value", "p.value", "sig", "adj.p.value", "adj.sig")]  
export_dir <- paste(here(),'/exports',sep='')
write.csv(df_stats, file=paste(export_dir,'/pdp1_mixed_models_v1.1.csv', sep=''), row.names=FALSE)