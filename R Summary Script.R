#R SCRIPT FOR BRANCH AND AREA SUMMARY 
###EXAMPLE SCRIPT 
#Load tidyverse
library(tidyverse)

#FUNCTIONS
#branch function for total branch length and rows
branchfunction <- function(file) {
  a <- read_csv(file)
  print(file)
  sumbranch <- sum(a$'branch-distance')
  rows <- nrow(a)
  n <- data.frame(rows)
  o <- data.frame(sumbranch)
  merge <- merge(n,o)
}

#sum function for total area
sumfunction <- function(file) {
  a <- read_csv(file)
  print(file)
  sum <- sum(a$Area)
  data.frame(area=sum)
}

##############EXAMPLE 

#BRANCHES
#set working directory to skeleton folder
setwd('C:/Users/USER/Documents/PROJECTTITLE_Image_Analysis')

#find files containing branches in name
branches<- dir(pattern = "_branches")
branches
#lapply function
allbranch<-lapply(branches, branchfunction)
#appy file names to names
names(allbranch)<-branches
#bind_rows of file
result <- bind_rows(allbranch,.id = "file")
result
#remove extra part of file names
result$file <- str_replace(result$file, pattern = "_branches.csv", replacement ="")
#create new column of grouping for filename
result <- result %>%
  mutate(group = str_extract(result$file, pattern = "[A-Z]+")) 
result %>%
  mutate(filename = str_extract(result$file, pattern = "[A-Z]+\\d+")) -> result

result
#write .csv to file
write.csv(result,file = 'ALL_BRANCHES.csv')

#AREA
#################################
#find files containing _proparea in name
area <- dir(pattern = "_proparea")
#lapply function
all_area<-lapply(area, sumfunction)
#appy file names to names
names(all_area)<-area
#bind_rows of file
area_result <- bind_rows(all_area,.id = "file")
#remove extra part of file names
area_result$file <- str_replace(area_result$file, pattern = "_proparea.csv", replacement ="")

area_result
#create new column of grouping for filename
area_result <- area_result %>%
  mutate(group = str_extract(area_result$file, pattern = "[A-Z]+"))
area_result
write.csv(area_result,file = 'ALL_AREA.csv')


