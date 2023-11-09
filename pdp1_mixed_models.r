library(lmerTest)
library(tidyr)
library(tibble)
library(dplyr)

setwd('C://Users//szb37//My Drive//Projects//PDP1//codebase//')
source('C://Users//szb37//My Drive//Projects//PDP1//codebase//data//load_pdp1.r')
df <- load_pdp1()
keep <- c('pID','tp','measure','result')
df <- subset(df, select=c(keep))
res <- data.frame()

### Do models
for (this_measure in unique(df$measure)) {
  
  if(this_measure=='MTSCTAPC'){
    next # all values are 100
  }
  
  #this_measure='PALFAMS'
  model <- lmer('result~(1|pID)+tp', subset(df, measure==this_measure))
  coeffs <- rownames_to_column(data.frame(coef(summary(model))), var = 'tp')
  coeffs$measure <- this_measure
  res <- rbind(res, coeffs)
}

res$est <- round(res$Estimate, 2)
res$Estimate <- NULL
res$SE <- round(res$Std..Error, 2)
res$Std..Error <- NULL
res$p <- res$Pr...t..
res$Pr...t.. <- NULL
res$df <- round(res$df, 1)

res <- res %>%
  mutate(sig =
    ifelse(p<=0.001, '***',
    ifelse(p<=0.01, '**',
    ifelse(p<=0.05, '*', ''))))
res$p <- round(res$p,5)

res <- res[, c("measure", "tp", "est", "SE", "df", "t.value", "p", "sig")]  
export_dir <- 'C://Users//szb37//My Drive//Projects//PDP1//codebase//export results//'
write.csv(res, file=paste(export_dir,'pdp1_mixed_models_v1.csv', sep=''), row.names=FALSE)
