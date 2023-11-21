library(tidyr)
library(tibble)
library(dplyr)
library(Rfit)

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
  model <- rfit('result~tp', subset(df, measure==this_measure))
  coeffs <- rownames_to_column(data.frame(coef(summary(model))), var = 'tp')
  coeffs$measure <- this_measure
  res <- rbind(res, coeffs)
}

res$est <- round(res$Estimate, 2)
res$Estimate <- NULL
res$SE <- round(res$Std..Error, 2)
res$Std..Error <- NULL
res$p <- res$p.value
res$p.value <- NULL
#res$df <- round(res$df, 1)

res <- res %>%
  mutate(sig =
    ifelse(p<=0.001, '***',
    ifelse(p<=0.01, '**',
    ifelse(p<=0.05, '*', ''))))
res$p <- round(res$p,5)

res <- res[, c("measure", "tp", "est", "SE", "t.value", "p", "sig")]  
write.csv(
  res, 
  file='C://Users//szb37//My Drive//Projects//PDP1//codebase//export results//rfit.csv', 
  row.names=FALSE)
