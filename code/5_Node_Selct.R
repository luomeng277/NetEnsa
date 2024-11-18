setwd('/home/LuoM/code/pythonProject4/testdata/result/')
library(readxl)
library(dplyr)

data1 <- read.csv("alldata.csv",header = T)

first_column <- data1$taxon_names
data2 <- read.csv("eccase_m.csv",  colClasses = c("NULL", rep("numeric", ncol(read.csv("bccase_m.csv")) - 1)))
data3 <- read.csv("eccontrol_m.csv", colClasses = c("NULL", rep("numeric", ncol(read.csv("bccontrol_m.csv")) - 1)))

abs_diff <- abs(data2 - data3)
options (warn = -1)

# Combine taxon_names and abs_diff into a new data frame
combined_data <- cbind(data1$taxon_names, abs_diff)

# Sort the combined data by the second column (abs_diff)
sorted_combined_data <- combined_data[order(combined_data[, 2]), ]

# Select the taxon_names column from sorted_combined_data
selected_names <- sorted_combined_data[, 1]

# Filter selected_names based on the condition
# Thresholds are selected based on the actual results and the number of decisions, with 1/5 of the original number of nodes elected as the maximum limit
 selected_names <- selected_names[abs_diff > 0.2]
# #
selected_data <- data.frame(taxon_names = selected_names)
 # selected_data <- selected_data[complete.cases(selected_data), ]
 write.csv(selected_data, file = "ec.csv", row.names = FALSE, col.names = "species")
 # Select only the "taxon_names" column from df1
 df1_taxon_names <- selected_data[, "species", drop = FALSE]
 
 # Read the second CSV file
 df2 <- read.csv("alldata.csv")

 merged_df <- merge(df1_taxon_names, df2, by = "species")
 
 # Output the result to a new CSV file
 write.csv(merged_df, "ec_aden.csv", row.names = FALSE)
 



