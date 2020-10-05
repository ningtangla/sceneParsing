import pandas as pd

dataFrameList = pd.concat([pd.read_csv('sub' + str(subjectId+1) + '.txt', sep = ' ', header = None) for subjectId in range(10)])
dataFrameList.columns = ['imageIndex','groundTruthOfAnswerType','identificateAnswerType']

leagleData = dataFrameList[dataFrameList['identificateAnswerType'].isin([1,2])]
totalAcc = len(leagleData[leagleData['groundTruthOfAnswerType'] == leagleData['identificateAnswerType']])*1.0/len(leagleData)
