library(foreign)
infile <- 'NGIR6AFL.dta?dl=0'
data <- read.dta(infile)
data_labels <- attr(data,"var.labels")
combined_column_names <- paste(colnames(data), data_labels, sep=" | ")
colnames(data) <- combined_column_names
data_outfile <- paste(strsplit(infile, '.dta')[[1]][1], ".csv", sep="")
write.table(data, file=data_outfile, sep=",")