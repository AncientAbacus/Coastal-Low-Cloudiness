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
import geopandas as gpd
from shapely.geometry import Point
import geopandas as gpd
from shapely.geometry import Point, box
from random import uniform
from concurrent.futures import ThreadPoolExecutor
from tqdm.notebook import tqdm
import geopandas as gpd
from shapely.geometry import Point

def min_distance(point, lines):
    return lines.distance(point).min()

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
us = world[world['name']=='United States of America'].dissolve(by='name')
coastline = gpd.clip(gpd.read_file('ne_10m_coastline.shp'), us)
coastline = coastline.set_crs('EPSG:3087')
points_df = gpd.GeoDataFrame({
    'geometry': [
        Point(59.646   -151), 
        Point(-81.735374, 30.121735)]}, crs='EPSG:4326')
points_df = points_df.to_crs('EPSG:3087') # https://epsg.io/3087

points_df['min_dist_to_coast'] = points_df.geometry.apply(min_distance, args=(coastline,))

print(points_df)

