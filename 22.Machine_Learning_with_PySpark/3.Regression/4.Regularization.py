## Cars again
assembler = VectorAssembler(inputCols = [
    'mass', 'cyl', 'type_dummy', 'density_line', 'density_quad'
], outputCol = 'features')

cars = assembler.transform(cars)

## Cars: Linear Regression
# Fit
regression = LinearRegression(labelCol = 'consumption').fit(cars_train)

## Cars: Ridge Regression
ridge = LinearRegression(labelCol = 'consumption', elasticNetParam = 0, regParam = 0.1)
ridge.fit(cars_train)

## Cars: Lasso Regression
lasso = LinearRegression(labelCol = 'consumption', elasticNetParam = 1, regParam = 0.1)
lasso.fit(cars_train)





# --------------------------------------------------------------------------------------------------------- #
##                  Flight duration model: More features!                  ##
# Subset from the flights DataFrame:
# 
# +--------------------------------------------+--------+
# |features                                    |duration|
# +--------------------------------------------+--------+
# |(32,[0,3,11],[3465.0,1.0,1.0])              |351     |
# |(32,[0,1,13,17,21],[509.0,1.0,1.0,1.0,1.0]) |82      |
# |(32,[0,2,10,19,23],[542.0,1.0,1.0,1.0,1.0]) |82      |
# |(32,[0,1,11,16,30],[1989.0,1.0,1.0,1.0,1.0])|195     |
# |(32,[0,1,10,20,25],[415.0,1.0,1.0,1.0,1.0]) |65      |
# +--------------------------------------------+--------+
# only showing top 5 rows


from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit linear regression model to training data
regression = LinearRegression(labelCol='duration', elasticNetParam = 0, regParam = 0.1).fit(flights_train)

# Make predictions on testing data
predictions = regression.transform(flights_test)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol = 'duration').evaluate(predictions)
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)



#  output:
#     The test RMSE is 10.724966997932256
#     [0.0743384279272672,25.775191877668387,18.79220982810944,50.48619229210071,44.30108279701667,16.128205515942934,13.65324614839298,15.741634082847094,-14.92988672723959,1.6410664081804578,4.0570827301258365,6.8625530885736845,4.585201961549088,8.777749513845563,8.695488979055842,0.3568719272297695,0.07020272622638968,-0.15569089069238334,0.19810552587745595,0.20502376804131292,0.12106035497160066,-2.146728803140271,-2.251899874193858,-2.0349591400343017,-3.616304680257623,-4.106127867286387,-4.325503108102908,-4.4862676270572655,-4.2352550912579465,-3.943235331450326,-2.8946617591999635,-0.7498040531589405]

    






##                  Flight duration model: Regularisation!                  ##
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# Fit Lasso model (λ = 1, α = 1) to training data
regression = LinearRegression(labelCol = 'duration', regParam = 1, elasticNetParam = 1).fit(flights_train)

# Calculate the RMSE on testing data
rmse = RegressionEvaluator(labelCol = 'duration').evaluate(regression.transform(flights_test))
print("The test RMSE is", rmse)

# Look at the model coefficients
coeffs = regression.coefficients
print(coeffs)

# Number of zero coefficients
zero_coeff = sum([beta == 0 for beta in regression.coefficients])
print("Number of coefficients equal to 0:", zero_coeff)


# output:
#     The test RMSE is 11.221618112066176
#     [0.07326284332459325,0.26927242574175647,-4.213823507520847,23.31411303902282,16.924833465407964,-7.538366699625629,-5.04321753247765,-20.348693139176927,0.0,0.0,0.0,0.0,0.0,1.199161974782719,0.43548357163388335,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
#     Number of coefficients equal to 0: 22





##                  Flight duration model: Adding origin airport                  ##







##                  Interpreting coefficients                  ##




## ====================================================================================================== ##