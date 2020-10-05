import pandas as pd
import math

humanIdentificationDataFrameList = pd.concat([pd.read_csv('sub' + str(subjectId+1) + '.txt', sep = ' ', header = None) for subjectId in range(2)])
humanIdentificationDataFrameList.columns = ['imageIndex','groundTruthOfAnswerType','identificateAnswerType']

leagleData = humanIdentificationDataFrameList[humanIdentificationDataFrameList['identificateAnswerType'].isin([1,2])]

leagleData['correctTrial'] = (leagleData['groundTruthOfAnswerType'] == leagleData['identificateAnswerType'])
identificate = leagleData.groupby('imageIndex').sum()
trailsByImageIndex = leagleData.groupby('imageIndex').size()
acc = identificate['correctTrial']/trailsByImageIndex
#leagleData['acc'] = len(leagleData[leagleData['groundTruthOfAnswerType'] == leagleData['identificateAnswerType']])*1.0/len(leagleData)

imageEntropyDataFrame = pd.read_csv('../../informationContent_conditionalEntropy_humanPrior10001-15000.csv')
imageIndexSortByConditionEntropy = imageEntropyDataFrame[imageEntropyDataFrame['expt3'].isin(list(range(1,21)))]
numGroup = 3
numImage = len(imageIndexSortByConditionEntropy)
numImageEachGroup = int(math.ceil(numImage*1.0/numGroup))

groupedImageIndexes = [imageIndexSortByConditionEntropy['imageIndex'][i*numImageEachGroup:(i+1) * numImageEachGroup].values for i in range(numGroup - 1)] + \
                      [imageIndexSortByConditionEntropy['imageIndex'][(numGroup -1 ) * numImageEachGroup : ].values] 
accOnGroupByEntropy = [acc[acc.index.isin(groupedImageIndex)].mean() for groupedImageIndex in groupedImageIndexes]

print(accOnGroupByEntropy)
