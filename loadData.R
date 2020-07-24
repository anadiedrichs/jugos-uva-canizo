library(stringr)
library(readxl)
library(dplyr)

# normalizamos los datos entre 0 y 1
range01 <- function(x){(x-min(x))/(max(x)-min(x))}


load_dataset <- function()
{
  dataset <- read_excel("../data/Dataset para enfoque 2 y 3.xlsx")
  dataset <- dataset %>% filter(Origen != "CH")
  
  # quito el ppb al final del nombre de la columna
  
  colnames(dataset) <- str_extract(colnames(dataset), "[A-Z][a-z]*")
  
  dataset <- data.frame(dataset[,1], apply(dataset[-1],2,range01))
  
  return(dataset)
}

load_raw_data <- function()
{
  read_excel("../data/Dataset para enfoque 2 y 3.xlsx")
}