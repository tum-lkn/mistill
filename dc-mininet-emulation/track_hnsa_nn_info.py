from array import array
import json
import glob
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bcknd
from dataclasses import dataclass
import sys
import random

data = pd.read_csv("time_tracking_logs/node_tor0_link_ud_test2_gpu.log", sep='\s+')
data.info()
df_times = data[['times']]
mean_d = df_times.agg(np.mean)
median_d = df_times.agg(np.median)
perc5 = df_times.agg(lambda x: np.percentile(x,5))
perc95 = df_times.agg(lambda x: np.percentile(x,95))
print("HNSA/HLSA NN execution times")
print("mean: " + str(mean_d) + " median: " + str(median_d) + " percentile 5: " + str(perc5) + " percentile 95: " + str(perc95))