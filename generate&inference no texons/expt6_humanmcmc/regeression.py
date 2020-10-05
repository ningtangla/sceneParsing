import numpy as np
import pandas as pd
import math
import cvxpy as cp

def calGaussianLogLikelihood(truth, prediction, numData):
    distance = truth - prediction
    varianceGaussian = np.std(distance) ** 2
    print(varianceGaussian, distance)
    logLikelihood = - numData / 2 * np.log(2 * math.pi * varianceGaussian) - np.sum(np.power(distance, 2))/ (2 * varianceGaussian) 
    return logLikelihood

data = pd.read_csv('Regression.csv')
initialInformation = data['Initial Information'].values 
finalInformation = data['Final Information'].values 

X = initialInformation
YTrue = finalInformation
numData = len(X)

linearCoef = cp.Variable(1)
linearIntercept = cp.Variable(1)

distance = YTrue - linearCoef * X - linearIntercept
leastSquareLoss = cp.sum_squares(distance)

linearObjective = cp.Minimize(leastSquareLoss)
prob = cp.Problem(linearObjective)

result = prob.solve()

linearTruth = YTrue
linearPridict = linearCoef.value * X - linearIntercept.value
linearLogLikelihood = calGaussianLogLikelihood(linearTruth, linearPridict, numData)
print(linearLogLikelihood)


antiLogXConstant = cp.Variable(1)
antiLogYConstant = cp.Variable(1)
logLinearCoef = cp.Variable(1)
logLinearIntercept = cp.Variable(1)

distanceLogLinear = YTrue - logLinearCoef * cp.log(X + antiLogXConstant) - logLinearIntercept
leastSquareLossLogLinear = cp.sum_squares(distanceLogLinear)

lowBound = -np.min(X)
logLinearConstarins = [antiLogXConstant >= lowBound] 
logLinearObjective = cp.Minimize(leastSquareLossLogLinear)
probLogLinear = cp.Problem(logLinearObjective, logLinearConstarins)
resultLoglinear = probLogLinear.solve()

print(logLinearCoef.value, -np.min(X))




