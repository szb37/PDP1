library(here)

load_pdp1 <- function(){
  filepath = paste(here(),'/data/pdp1_MASTER_v1.csv', sep ='')
  df <- read.csv(filepath, header=TRUE, sep = ",", stringsAsFactors=FALSE)
  
  df$gender <- as.factor(df$gender)
  df$edu <- as.factor(df$edu)
  df$tp <- as.factor(df$tp)
  df$tp <- factor(df$tp, levels=c('bsl', 'A1', 'A7', 'B1', 'B7', 'B30', 'B90'))
  df$edu <- relevel(df$edu, ref = "LOE_3")
  df$tp <- relevel(df$tp, ref = "bsl")
  
  return(df)
}


