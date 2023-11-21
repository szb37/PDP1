rm(list = ls())
library(dplyr)
library(tidyr)
library(tibble)

setwd('C://Users//szb37//My Drive//Projects//PDP1//codebase//')
source('C://Users//szb37//My Drive//Projects//PDP1//codebase//data//load_pdp1.r')
df <- load_pdp1()
keep <- c('pID','tp','measure','result')
df <- subset(df, select=c(keep))

df <- pivot_wider(
  df,
  id_cols=c('pID', 'measure'),
  names_from = 'tp',
  values_from='result')

res <- tibble(
  measure = character(),
  tp = character(),
  t.stat = numeric(),
  t.df = numeric(),
  t.p = numeric(),
  wilcox.stat = numeric(),
  #wilcox.df = numeric(),
  wilcox.p = numeric(),
  hedges.g = numeric(),)

calc_hedges.g <- function(group1, group2) {
  # Calculate means
  mean1 <- mean(group1)
  mean2 <- mean(group2)
  
  # Calculate pooled standard deviation
  pooled_sd <- sqrt(((length(group1) - 1) * var(group1) + (length(group2) - 1) * var(group2)) / (length(group1) + length(group2) - 2))
  
  # Calculate Hedge's g
  hedges_g <- (mean1 - mean2) / pooled_sd
  
  return(hedges_g)
}

### Do tests
for (this_measure in unique(df$measure)) {
  
  if(this_measure=='MTSCTAPC'){
    next # all values are 100
  }
  
  for (this_tp in c('A7', 'B7', 'B30')) {
    
    ### Wrangling
    tmp_df <- subset(df, measure==this_measure)
    keep <- c('pID','measure','bsl', this_tp)
    tmp_df <- subset(tmp_df, select=c(keep))
    if (this_tp=='A7'){
      colnames(tmp_df)[colnames(tmp_df)=="A7"] <- "ep"
    } else if (this_tp=='B7'){
      colnames(tmp_df)[colnames(tmp_df)=="B7"] <- "ep"
    } else if (this_tp=='B30'){
      colnames(tmp_df)[colnames(tmp_df)=="B30"] <- "ep"
    }
    
    ### Do tests
    wilcox <- wilcox.test(tmp_df$bsl, tmp_df$ep, paired=TRUE)
    ttest <- t.test(tmp_df$bsl, tmp_df$ep, paired=TRUE)
    ### Calc ES
    g <- calc_hedges.g(tmp_df$bsl, tmp_df$ep)
    
    ### Add new row to res
    new_row <- tibble(
      measure = this_measure,
      tp = this_tp,
      t.stat = round(ttest$statistic,3),
      t.df = round(ttest$parameter,3),
      t.p = round(ttest$p.value,5),
      wilcox.stat = round(wilcox$statistic,3),
      #wilcox.df = round(wilcox$parameter,3),
      wilcox.p = round(wilcox$p.value,5),
      hedges.g = round(g,3))
    
    res <- bind_rows(res, new_row)
  }
}

### Significance formatting
res <- res %>%
  mutate(t.sig = 
    ifelse(t.p<=0.001, '***', 
    ifelse(t.p<=0.01, '**', 
    ifelse(t.p<=0.05, '*', ''))))

res <- res %>%
  mutate(wilcox.sig = 
    ifelse(wilcox.p<=0.001, '***', 
    ifelse(wilcox.p<=0.01, '**', 
    ifelse(wilcox.p<=0.05, '*', ''))))


export_dir <- 'C://Users//szb37//My Drive//Projects//PDP1//codebase//export results//'
write.csv(res, file=paste(export_dir,'pdp1_wilcoxon_ttests_v1.csv', sep=''), row.names=FALSE)