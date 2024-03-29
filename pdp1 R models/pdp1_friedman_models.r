rm(list = ls())
library(tidyr)
library(dplyr)
library(PMCMRplus)
library(here)

source(paste(here(),'/pdp1 R models/pdp1_helpers.r', sep=''))
df <- load_pdp1_data()

measures <- c(
  'UPDRS_1', 'UPDRS_2', 'UPDRS_3', 'UPDRS_4', 'UPDRS_SUM',
  'HAMA', 'MADRS', 'NPIQ_DIS', 'NPIQ_SEV',  'CSSRS', 'ESAPS', 'CCFQ',
  'PRL', 'Z_MTS', 'Z_OTS', 'Z_PAL', 'Z_RTI', 'Z_SWM',
  'MTSCFAPC', 'MTSPS82', 'MTSRCAMD', 'MTSRFAMD', 'OTSMDLFC', 'OTSPSFC',
  'PALFAMS', 'PALTEA', 'RTIFMDMT', 'RTIFMDRT', 'RTISMDMT', 'RTISMDRT',
  'SWMBE12', 'SWMBE4', 'SWMBE468', 'SWMBE6', 'SWMBE8', 'SWMS')

### Initalize outputs
column_names <- c("measure", "stat", "p.value")
df_main <- data.frame(matrix(ncol = length(column_names)))
colnames(df_main) <- column_names

column_names <- c("measure", "tp", "stat", "p.value")
df_pw <- data.frame(matrix(ncol = length(column_names)))
colnames(df_pw) <- column_names


### Do Friedman models
for (this_measure in measures) {
  print(this_measure)
  
  tmp <- subset(df, (measure==this_measure) & (tp %in% c('bsl', 'A7', 'B7', 'B30')))
  tmp$tp <- factor(tmp$tp, levels=c('bsl', 'A7', 'B7', 'B30'))
  
  # Delete incomplete pIDs if needed
  if (this_measure %in% c('RTISMDMT', 'RTISMDRT')){
    tmp <- tmp[tmp$pID!=1051,] 
    }
  else if (this_measure=='PRL') {
    tmp <- tmp[tmp$pID!=1020,] 
    tmp <- tmp[tmp$pID!=1083,] 
    tmp <- tmp[tmp$pID!=1085,] 
    tmp <- tmp[tmp$pID!=1129,] 
    tmp <- tmp[tmp$pID!=1145,] 
  }
  else if (this_measure=='UPDRS_4') {
    tmp <- tmp[tmp$pID!=1142,] 
  }
  else if (this_measure=='UPDRS_SUM') {
    tmp <- tmp[tmp$pID!=1142,] 
  }
  
  ### Get omnibus test results
  friedman <- friedman.test(score ~ tp|pID, tmp)
  coeffs <- c(this_measure, friedman$statistic, round(friedman$p.value, 3))
  df_main <- rbind(df_main, coeffs)
  
  ### Get pairwise comparisons
  pairwise <- frdAllPairsExactTest(tmp$score, tmp$tp, tmp$pID, p.adjust.method="none")
  coeffs_A7 <-  c(this_measure, 'A7', round(pairwise$statistic['A7', 'bsl'],3), round(pairwise$p.value['A7', 'bsl'],3))
  coeffs_B7 <-  c(this_measure, 'B7', round(pairwise$statistic['B7', 'bsl'],3), round(pairwise$p.value['B7', 'bsl'],3))
  coeffs_B30 <- c(this_measure, 'B30', round(pairwise$statistic['B30', 'bsl'],3), round(pairwise$p.value['B30', 'bsl'],3))
  df_pw <- rbind(df_pw, coeffs_A7)
  df_pw <- rbind(df_pw, coeffs_B7)
  df_pw <- rbind(df_pw, coeffs_B30)
}


### Format outputs
df_main <- na.omit(df_main)
df_pw <- na.omit(df_pw)

df_main <- df_main %>%
  mutate(sig =
    ifelse(p.value<=0.001, '***',
    ifelse(p.value<=0.01, '**',
    ifelse(p.value<=0.05, '*', ''))))
df_main$p.value <- ifelse(df_main$p.value<0.001, '<0.001', df_main$p.value)

df_pw <- df_pw %>%
  mutate(sig =
    ifelse(p.value<=0.001, '***',
    ifelse(p.value<=0.01, '**',
    ifelse(p.value<=0.05, '*', ''))))
df_pw$p.value <- ifelse(df_pw$p.value<0.001, '<0.001', df_pw$p.value)

export_dir <- paste(here(),'exports',sep='/')
write.csv(df_main, file=paste(export_dir,'pdp1_friedman_main.csv', sep='/'), row.names=FALSE)
write.csv(df_pw, file=paste(export_dir,'pdp1_friedman_pairwise.csv', sep='/'), row.names=FALSE)