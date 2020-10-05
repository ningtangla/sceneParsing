import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

data = pd.read_csv('Regression.csv')
initialInformation = data['Initial Information'].values 
finalInformation = data['Final Information'].values 

linearPredictY = sm.add_constant(initialInformation)
tureY = finalInformation 

linearModel = sm.OLS(tureY, linearPredictY)
linearResults = linearModel.fit()
print(linearResults.summary())

antiLog = sm.add_constant(initialInformation)
logX = np.log(antiLog)
logLinearPredictY = sm.add_constant(logX)
logLinearTrueY = sm.add_constant(trueY)

logLineaModel = sm.GLM(tureY, logLinearPredictY)
logLinearResults = logLineaModel.fit_constrained()#unable to do inequallity constrain
print(logLinearResults.summary())

