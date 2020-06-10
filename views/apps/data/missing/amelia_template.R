

# Install Amelia if we don't have it
if (!require("Amelia")) install.packages("Amelia", repos="https://ftp.acc.umu.se/mirror/CRAN/")
library("Amelia")

library("foreign")
library("methods")
library("parallel")

find_bounds <- function(df){
  print("finding bounds")
  lower <- c()
  upper <- c()
  for (i in 1:length(df)) {
    lower <- c(lower, min(df[,i], na.rm=T))
    upper <- c(upper, max(df[,i], na.rm=T))
  }

  varnr <- c(1:ncol(df))
  lower <- lower[varnr]
  upper <- upper[varnr]
  bounds <- matrix(cbind(varnr,lower,upper),ncol(df))

  return(bounds)

}

keep_only_varying <- function(df){
  print("Removing non-varying columns from dataframe")
  # Find variance to remove non-varying variables.
  variances <- sapply(df, var, na.rm = TRUE)

  # Some vars are all missing, they get variance NA, give them zero instead
  variances[is.na(variances)] <- 0

  names.zero.variance <- colnames(df[variances == 0])
  names.positive.variance <- colnames(df[variances > 0])
  df <- df[names.positive.variance]

  return(df)
}

keep_only_numerics <- function(df){
  print("Removing non-numeric columns from dataframe")
  numerics <- sapply(df, is.numeric)
  df <- df[numerics]

  return(df)
}


time_start <- Sys.time()
print("Starting amelia imputation script")

path_csv_input <- "${PATH_CSV_INPUT}"
path_csv_output_stem <- "${PATH_CSV_OUTPUT_STEM}"
timevar <- "${TIMEVAR}"
groupvar <- "${GROUPVAR}"
n_imp <- ${N_IMP}
n_cpus <- ${N_CPUS}

print(paste("path_csv_input", path_csv_input))
print(paste("path_csv_output_stem", path_csv_output_stem))
print(paste("timevar", timevar))
print(paste("groupvar", groupvar))
print(paste("n_imp", n_imp))
print(paste("n_cpus", n_cpus))

df <- read.csv(path_csv_input)

# # Drop all vars that don't vary
# df <- keep_only_varying(df)
# # Drop all non-numeric vars
# df <- keep_only_numerics(df)

nominals <- c()

# Find the bounds of each var, we don't want never-before seen values
bounds <- find_bounds(df)

# Run the imputation
obj_amelia <- amelia(df,
                     m = n_imp,
                     ts = timevar,
                     cs = groupvar,
                     noms = nominals,
                     p2s = 2,
                     polytime = 1,
                     intercs = TRUE,
                     empri = .1*nrow(df),
                     bounds = bounds,
                     max.resample = 1000,
                     parallel = "multicore",
                     ncpus = n_cpus)

print("Finished imputing")

write.amelia(obj=obj_amelia,
             file.stem = path_csv_output_stem, format = "csv")
print("Saved imputed datasets")

time_end <- Sys.time()
time_total = time_end - time_start
print("FINISHED!")
print(paste("Total runtime:", time_total))

