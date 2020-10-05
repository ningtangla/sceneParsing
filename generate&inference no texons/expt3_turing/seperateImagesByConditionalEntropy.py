import pandas as pd
import numpy as np 

informationTheoryDf = pd.read_csv('informationcontent_conditionalentropy_human_prior.csv')
treeConditionOnImageEntropy = informationTheoryDf['conditional_entropy']
sortedImageIndex = np.argsort(treeConditionOnImageEntropy.values) + 1

entropyConditions = ['lowEntropy', 'highEntropy']
numLevelsToSeperateImagesByEntropy = len(entropyConditions)
numImagesEachLevel = int(len(sortedImageIndex) / numLevelsToSeperateImagesByEntropy) - 11
imageIndexsByLevel = [sortedImageIndex[(-levelIndex) * numImagesEachLevel : (1-levelIndex) * numImagesEachLevel -1] for levelIndex in range(numLevelsToSeperateImagesByEntropy)]

modelConditions = ['humanPrior','uniformPrior','randomSelect']

correctIdentifyByModel = []
for modelCondition in modelConditions:
    modelIndex = modelConditions.index(modelCondition)
    humanDataDf = pd.read_excel('turing1-' + str(modelIndex) + '.xlsx', sheetname = 'turing1-' + str(modelIndex))
    numCorrectIdentifyByLevel = [len(humanDataDf[np.isin(humanDataDf.img, imageIndexList) & (humanDataDf.Acc != 1)]) for imageIndexList in imageIndexsByLevel]
    numImagesByLevel = [len(humanDataDf[np.isin(humanDataDf.img, imageIndexList)]) for imageIndexList in imageIndexsByLevel]
    correctIdentifyPercentByLevel = np.array(numCorrectIdentifyByLevel) / np.array(numImagesByLevel)
    print(correctIdentifyPercentByLevel)
    correctIdentifyByModel.append(correctIdentifyPercentByLevel)

__import__('ipdb').set_trace()
