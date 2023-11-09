rm(list = ls())
library(tidyr)
library(dplyr)
library(PMCMRplus)

setwd('C://Users//szb37//My Drive//Projects//PDP1//codebase//')
source('C://Users//szb37//My Drive//Projects//PDP1//codebase//data//load_pdp1.r')
df <- load_pdp1(rm.extreme=TRUE)
keep <- c('pID','tp','measure','result')
df <- subset(df, select=c(keep))

column_names <- c("measure", "stat", "p.value")
res_main <- data.frame(matrix(ncol = length(column_names)))
colnames(res_main) <- column_names

column_names <- c("measure", "tp", "stat", "p.value")
res_pw <- data.frame(matrix(ncol = length(column_names)))
colnames(res_pw) <- column_names

### Do Friedman models
for (this_measure in unique(df$measure)) {
  
  if(is.element(this_measure, c('MTSCTAPC'))){
    next # all values are 100
  }
  
  friedman <- friedman.test(result ~ tp|pID, subset(df, measure==this_measure))
  coeffs <- c(this_measure, round(friedman$statistic,3), round(friedman$p.value,5))
  res_main <- rbind(res_main, coeffs)
  
  
  nemenyi <- frdAllPairsNemenyiTest(result ~ tp|pID, subset(df, measure==this_measure))
  coeffs_A7 <-  c(this_measure, 'A7', round(nemenyi$statistic['A7', 'bsl'],3), round(nemenyi$p.value['A7', 'bsl'],5))
  coeffs_B7 <-  c(this_measure, 'B7', round(nemenyi$statistic['B7', 'bsl'],3), round(nemenyi$p.value['B7', 'bsl'],5))
  coeffs_B30 <- c(this_measure, 'B30', round(nemenyi$statistic['B30', 'bsl'],3), round(nemenyi$p.value['B30', 'bsl'],5))
  res_pw <- rbind(res_pw, coeffs_A7)
  res_pw <- rbind(res_pw, coeffs_B7)
  res_pw <- rbind(res_pw, coeffs_B30)
}

res_main <- na.omit(res_main)
res_main <- res_main %>%
  mutate(sig =
    ifelse(p.value<=0.001, '***',
    ifelse(p.value<=0.01, '**',
    ifelse(p.value<=0.05, '*', ''))))

res_pw <- na.omit(res_pw)
res_pw <- res_pw %>%
  mutate(sig =
    ifelse(p.value<=0.001, '***',
    ifelse(p.value<=0.01, '**',
    ifelse(p.value<=0.05, '*', ''))))

export_dir <- 'C://Users//szb37//My Drive//Projects//PDP1//codebase//export results//'
write.csv(res_main, file=paste(export_dir,'pdp1_friedman_main_v1.csv', sep=''), row.names=FALSE)
write.csv(res_pw, file=paste(export_dir,'pdp1_friedman_pw_v1.csv', sep=''), row.names=FALSE)

### Toy example
# data <- data.frame(
#   Subject = rep(1:8, each = 3),
#   Time = rep(1:3, times = 8),
#   Response = c(
#     10, 9, 6, # subject 1 tp 1-3
#     11, 8, 7, 
#     10, 7, 6, 
#     13, 9, 5, 
#     10, 10, 5,
#     11, 7, 8, 
#     10, 8, 7, 
#     13, 9, 6)
# )
# res <- friedman.test(Response ~ Time | Subject, data = data)
# pw_res <- frdAllPairsNemenyiTest(Response ~ Time | Subject, data = data)
#
# res <- friedman.test(result ~ tp|pID, subset(df, measure==this_measure))
# pw_res <- frdAllPairsNemenyiTest(result ~ tp|pID, subset(df, measure==this_measure))
#
# if(is.element(this_measure, c('MTSCTAPC', 'RTISMDRT', 'RTISMDMT'))){
#   next # all values are 100
# }

# this_measure="PALTEA"
# this_measure="RTISMDRT"