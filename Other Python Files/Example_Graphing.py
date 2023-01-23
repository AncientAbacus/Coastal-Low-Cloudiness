import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import pandas as pd
import quandl
import pytz
import math
import os
import glob
import scipy.stats
from scipy import stats
import pandas as pd , numpy as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sma
import seaborn as sns
from valid import *
import itertools

t = pd.read_csv('MonLCfl_monthlyLowCloudFreq_AP20_7101316PST_wNA_090413_txt_111022.txt', sep = " ", header = 0, index_col = 0)

x = []
y = range(1950,2013)

t['1_average'] = t.mean(axis=1)

for col in t.columns:
	t[col].replace(to_replace = "nan", value = np.nan, inplace = True)

t = t.groupby(np.arange(len(t))//5).mean()

t.index = list(range(1950,2013))

t = t.reindex(sorted(t.columns), axis=1)

t.to_csv('Rachel_CLCs.csv', sep='\t')

fig, ax = plt.subplots()

plt.plot(y, t['1_average'], color='blue', marker='o', fillstyle='none', linestyle='-', linewidth=2, markersize=5)

plt.ylabel("Pacific Rim Avg CLC")

plt.xlabel("1950:2022")

plt.legend(["Summer CLC"], title='Legend', loc="upper left", bbox_to_anchor=(1.05,0.8))

ax.xaxis.grid(True, which='major')
ax.yaxis.grid(True, which='major')

plt.title("EX Pacific Rim Avg")

plt.savefig("EX Pacfic Rim Graph.pdf", dpi=300, format='pdf', bbox_inches='tight')

plt.show()

plt.clf()

plt.close()
