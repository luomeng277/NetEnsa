setwd('/home/LuoM/code/pythonProject4/testdata/result/')
library(readxl)
library(dplyr)
# Read the CSV file
df1 <- read.csv(paste0("alldata.csv"))

# Select only the "taxon_names" column from df1
df1_species_names <- df1[, "species", drop = FALSE]

# Read the second CSV file
df2 <- read.csv("ensa_sel.csv")

merged_df <- merge(df1_species_names, df2, by = "species")



# Output the result to a new CSV file
write.csv(merged_df, "aden.csv", row.names = FALSE)

