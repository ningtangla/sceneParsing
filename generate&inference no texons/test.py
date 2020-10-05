import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

data = sm.datasets.spector.load_pandas()
exog = data.exog
endog = data.endog
__import__('ipdb').set_trace()
print(data)
print(sm.datasets.spector.NOTE)
print(data.exog.head())

