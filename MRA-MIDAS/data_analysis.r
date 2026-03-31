# download data
library(readxl)
data <- read_excel("MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx")

View(data)

data_classified <- data[!is.na(data$midas_path), ]
data_unclassified <- data[is.na(data$midas_path), ]


# benign = 0, malignant = 1
data_classified$binary_label <- ifelse(grepl("malignant", data_classified$midas_path, ignore.case = TRUE), 1, 0)

View(data_classified)

benign <- data_classified[data_classified$binary_label == 0, ]
malignant <- data_classified[data_classified$binary_label == 1, ]


print(paste("number of benign patients:", nrow(benign)))
print(paste("number of malignant patients:", nrow(malignant)))
print(paste("number of unclassified patients:", nrow(data_unclassified)))
print(paste("number of patients (all):", nrow(data)))



locations <- unique(data_classified$midas_location)
print(paste("number of different locations", length(locations)))

location_df <- data.frame("count_classified" = integer(length(locations)),
                          "count_unclassified" = integer(length(locations)),
                          row.names = locations)


for (i in seq_along(locations)) {
  loc <- locations[i]
  location_data_classified <- data_classified[data_classified$midas_location == loc, ]
  location_data_unclassified <- data_unclassified[data_unclassified$midas_location == loc, ]

  location_df$count_classified[i] <- nrow(location_data_classified)
  location_df$count_unclassified[i] <- nrow(location_data_unclassified)
}

View(location_df)
write.csv(location_df)
table(location_df$count_classified)

print(paste("Number of patients:", length(unique(data$midas_record_id))))
