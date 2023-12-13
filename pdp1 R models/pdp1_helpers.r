library(here)

load_pdp1_data <- function(){
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

calc_hedges.g <- function(group1, group2) {
  
  # Remove NAs
  group1 <- na.omit(group1)
  group2 <- na.omit(group2)
  
  # Calculate means
  mean1 <- mean(group1)
  mean2 <- mean(group2)
  
  # Calculate pooled standard deviation
  pooled_sd <- sqrt(((length(group1) - 1) * var(group1) + (length(group2) - 1) * var(group2)) / (length(group1) + length(group2) - 2))
  
  # Calculate Hedge's g
  hedges_g <- round((mean1-mean2) / pooled_sd,2)
  
  return(hedges_g)
}

format_mixedmodel_output <- function(df){
  df$est <- round(df$Estimate, 2)
  df$Estimate <- NULL
  df$SE <- round(df$Std..Error, 2)
  df$Std..Error <- NULL
  df$p.value <- df$Pr...t..
  df$Pr...t.. <- NULL
  df$df <- round(df$df, 1)
  df$t.value <- round(df$t.value, 2)
  return(df)
}

rename_timepoints <- function(df){
  df$tp <- sub('(Intercept)', 'Intercept', df$tp)
  df$tp <- sub('tpA1',  'tp(A1)', df$tp)
  df$tp <- sub('tpA7',  'tp(A7)', df$tp)
  df$tp <- sub('tpB1',  'tp(B1)', df$tp)
  df$tp <- sub('tpB7',  'tp(B7)', df$tp)
  df$tp <- sub('tpB30', 'tp(B30)', df$tp)
  df$tp <- sub('tpB90', 'tp(B90)', df$tp)
  return(df)
}

add_sig_stars <- function(df){
  # Assumes DF has columns p.value and adj.p.value
  
  df <- df %>%
    mutate(sig =
      ifelse(p.value<=0.001, '***',
      ifelse(p.value<=0.01, '**',
      ifelse(p.value<=0.05, '*', ''))))
  
  df <- df %>%
    mutate(adj.sig =
      ifelse(adj.p.value<=0.001, '***',
      ifelse(adj.p.value<=0.01, '**',
      ifelse(adj.p.value<=0.05, '*', ''))))

  df <- df %>%
    mutate(p.value = ifelse(
      p.value<=0.001, '<0.001',
      format(round(df$p.value,3), nsmall=3)))
  
  df <- df %>%
    mutate(adj.p.value = ifelse(
      adj.p.value<=0.001, '<0.001',
      format(round(df$adj.p.value, 3), nsmall=3)))
  
  return(df)
}