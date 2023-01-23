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

def check_valid(year_frame):

    # filter valid observations at hour 7, 10, 13, 16
    all_s = year_frame[(year_frame.index.month >= 5) & (year_frame.index.month <= 9)] 
    all_s = all_s[(all_s["cld.base.ft"] != -999) & (all_s["cld.frac"] != -999.9)]
    # if there is no cloud height for cloud fraction >=0.75, we consider the observation not valid
    all_s = all_s[(((all_s["cld.frac"] >= 0.75) & (all_s["cld.base.ft"] != 72200)) | (all_s["cld.frac"] < 0.75))]
    # only looking at these hours
    # CURRENTLY NOT IN USE
    all_s = all_s[(all_s.index.hour == 7) | (all_s.index.hour == 10) | (all_s.index.hour == 13) | (all_s.index.hour == 16)]

    # check if hour 7 observations %s are valid
    all_s_seven = all_s[all_s.index.hour == 7]

    all_s_seven_may = all_s_seven[all_s_seven.index.month == 5]
    all_s_seven_june = all_s_seven[all_s_seven.index.month == 6]
    all_s_seven_july = all_s_seven[all_s_seven.index.month == 7]
    all_s_seven_august = all_s_seven[all_s_seven.index.month == 8]
    all_s_seven_sept = all_s_seven[all_s_seven.index.month == 9]

    if int(all_s_seven_may[all_s_seven_may.columns[0]].count())/31 >= 0.25:
        valid_seven_may = True
    else: valid_seven_may = False
    if int(all_s_seven_june[all_s_seven_june.columns[0]].count())/30 >= 0.25:
        valid_seven_june = True
    else: valid_seven_june = False
    if int(all_s_seven_july[all_s_seven_july.columns[0]].count())/31 >= 0.25:
        valid_seven_july = True
    else: valid_seven_july = False
    if int(all_s_seven_august[all_s_seven_august.columns[0]].count())/31 >= 0.25:
        valid_seven_august = True
    else: valid_seven_august = False
    if int(all_s_seven_sept[all_s_seven_sept.columns[0]].count())/30 >= 0.25:
        valid_seven_sept = True
    else: valid_seven_sept = False
    valid_seven = valid_seven_may and valid_seven_june and valid_seven_july and valid_seven_august and valid_seven_sept

    # check if hour 10 observations %s are valid
    all_s_ten = all_s[all_s.index.hour == 10]

    all_s_ten_may = all_s_ten[all_s_ten.index.month == 5]
    all_s_ten_june = all_s_ten[all_s_ten.index.month == 6]
    all_s_ten_july = all_s_ten[all_s_ten.index.month == 7]
    all_s_ten_august = all_s_ten[all_s_ten.index.month == 8]
    all_s_ten_sept = all_s_ten[all_s_ten.index.month == 9]

    if int(all_s_ten_may[all_s_ten_may.columns[0]].count())/31 >= 0.25:
        valid_ten_may = True
    else: valid_ten_may = False
    if int(all_s_ten_june[all_s_ten_june.columns[0]].count())/30 >= 0.25:
        valid_ten_june = True
    else: valid_ten_june = False
    if int(all_s_ten_july[all_s_ten_july.columns[0]].count())/31 >= 0.25:
        valid_ten_july = True
    else: valid_ten_july = False
    if int(all_s_ten_august[all_s_ten_august.columns[0]].count())/31 >= 0.25:
        valid_ten_august = True
    else: valid_ten_august = False
    if int(all_s_ten_sept[all_s_ten_sept.columns[0]].count())/30 >= 0.25:
        valid_ten_sept = True
    else: valid_ten_sept = False
    valid_ten = valid_ten_may and valid_ten_june and valid_ten_july and valid_ten_august and valid_ten_sept

    # check if hour 13 observations %s are valid
    all_s_thirteen = all_s[all_s.index.hour == 13]

    all_s_thirteen_may = all_s_thirteen[all_s_thirteen.index.month == 5]
    all_s_thirteen_june = all_s_thirteen[all_s_thirteen.index.month == 6]
    all_s_thirteen_july = all_s_thirteen[all_s_thirteen.index.month == 7]
    all_s_thirteen_august = all_s_thirteen[all_s_thirteen.index.month == 8]
    all_s_thirteen_sept = all_s_thirteen[all_s_thirteen.index.month == 9]

    if int(all_s_thirteen_may[all_s_thirteen_may.columns[0]].count())/31 >= 0.25:
        valid_thirteen_may = True
    else: valid_thirteen_may = False
    if int(all_s_thirteen_june[all_s_thirteen_june.columns[0]].count())/30 >= 0.25:
        valid_thirteen_june = True
    else: valid_thirteen_june = False
    if int(all_s_thirteen_july[all_s_thirteen_july.columns[0]].count())/31 >= 0.25:
        valid_thirteen_july = True
    else: valid_thirteen_july = False
    if int(all_s_thirteen_august[all_s_thirteen_august.columns[0]].count())/31 >= 0.25:
        valid_thirteen_august = True
    else: valid_thirteen_august = False
    if int(all_s_thirteen_sept[all_s_thirteen_sept.columns[0]].count())/30 >= 0.25:
        valid_thirteen_sept = True
    else: valid_thirteen_sept = False
    valid_thirteen = valid_thirteen_may and valid_thirteen_june and valid_thirteen_july and valid_thirteen_august and valid_thirteen_sept

    # check if hour 16 observations %s are valid
    all_s_sixteen = all_s[all_s.index.hour == 16]

    all_s_sixteen_may = all_s_sixteen[all_s_sixteen.index.month == 5]
    all_s_sixteen_june = all_s_sixteen[all_s_sixteen.index.month == 6]
    all_s_sixteen_july = all_s_sixteen[all_s_sixteen.index.month == 7]
    all_s_sixteen_august = all_s_sixteen[all_s_sixteen.index.month == 8]
    all_s_sixteen_sept = all_s_sixteen[all_s_sixteen.index.month == 9]

    if int(all_s_sixteen_may[all_s_sixteen_may.columns[0]].count())/31 >= 0.25:
        valid_sixteen_may = True
    else: valid_sixteen_may = False
    if int(all_s_sixteen_june[all_s_sixteen_june.columns[0]].count())/30 >= 0.25:
        valid_sixteen_june = True
    else: valid_sixteen_june = False
    if int(all_s_sixteen_july[all_s_sixteen_july.columns[0]].count())/31 >= 0.25:
        valid_sixteen_july = True
    else: valid_sixteen_july = False
    if int(all_s_sixteen_august[all_s_sixteen_august.columns[0]].count())/31 >= 0.25:
        valid_sixteen_august = True
    else: valid_sixteen_august = False
    if int(all_s_sixteen_sept[all_s_sixteen_sept.columns[0]].count())/30 >= 0.25:
        valid_sixteen_sept = True
    else: valid_sixteen_sept = False
    valid_sixteen = valid_sixteen_may and valid_sixteen_june and valid_sixteen_july and valid_sixteen_august and valid_sixteen_sept

    # return indices: 2 = May, 3 = June, 4 = July, 5 = August, 6 = Sept
    return valid_seven&valid_ten&valid_thirteen&valid_sixteen, all_s, valid_seven_may&valid_ten_may&valid_thirteen_may&valid_sixteen_may, valid_seven_june&valid_ten_june&valid_thirteen_june&valid_sixteen_june, valid_seven_july&valid_ten_july&valid_thirteen_july&valid_sixteen_july, valid_seven_august&valid_ten_august&valid_thirteen_august&valid_sixteen_august, valid_seven_sept&valid_ten_sept&valid_thirteen_sept&valid_sixteen_sept


