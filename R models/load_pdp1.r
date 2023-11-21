load_pdp1 <- function(rm.extreme=TRUE){
  df <- read.csv("C://Users//szb37//My Drive//Projects//PDP1//codebase//data//pdp1_v2.csv", header=TRUE, sep = ",", stringsAsFactors=FALSE)
  df$gender <- as.factor(df$gender)
  df$edu <- as.factor(df$edu)
  df$tp <- as.factor(df$tp)
  df$test <- as.factor(df$test)
  df$measure <- as.factor(df$measure)
  
  df$edu <- relevel(df$edu, ref = "LOE_3")
  df$tp <- relevel(df$tp, ref = "bsl")
  
  if(rm.extreme==TRUE){
    ### Selection of extreme values is based on Ellen's presentation
    extremes <- subset(df, pID=='PDP1001051' & is.element(df$measure, c('RTISMDMT','RTISMDRT')))
    extremes_idx <- as.numeric(c(row.names(extremes)))
    df <- df[-extremes_idx,]
  }
  
  return(df)
}


