library("EBMAforecast")

# These are templated values
path_calib_actuals <- "${PATH_CSV_ACTUALS}"
path_csv_calib<- "${PATH_CSV_CALIB}"
path_csv_test <- "${PATH_CSV_TEST}"
path_ebma <- "${PATH_CSV_EBMA}"
path_weights <- "${PATH_CSV_WEIGHTS}"



y_actual <- read.csv(path_calib_actuals, header=TRUE)
df_calib <- read.csv(path_csv_calib, header=TRUE)
df_test = read.csv(path_csv_test, header=TRUE)

colnames <- c(colnames(df_test))

# Equal weights by default
n_models = ncol(df_calib)
initial_weights <- rep((1/n_models), times=n_models)
# logit, normal, binary
param_model <- "logit"

# Defaults
param_tolerance <- ${PARAM_TOLERANCE}
param_shrinkage <- ${PARAM_SHRINKAGE}
param_const <- ${PARAM_CONST}
param_maxiter <- ${PARAM_MAXITER}

print("Started making forecast data")
fd <- EBMAforecast::makeForecastData(
    .predCalibration=df_calib,
    .predTest=df_test,
    .outcomeCalibration=y_actual$$actual, #double $$ for template
    .modelNames=colnames
    )

print("Started calibrateEnsemble")
ebma <- EBMAforecast::calibrateEnsemble(
    fd,
    model=param_model,
    tol=param_tolerance,
    exp=param_shrinkage,
    const=param_const,
    W=initial_weights,
    maxIter=param_maxiter
    )


ebma_prediction <- ebma@predTest[, "EBMA", ]
weights <- ebma@modelWeights

print("Writing result csvs")
write.csv(weights, path_weights, row.names=FALSE)
write.csv(ebma_prediction, path_ebma, row.names=FALSE)