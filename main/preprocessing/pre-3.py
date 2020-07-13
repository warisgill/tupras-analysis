# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### combining alarms into 1 csv 

# %%
import numpy as np
import json
import pandas as pd
from datetime import datetime
from dateutil.parser import parse


# %%
f0 = "formatted-pre-2-march2019.csv"
f1 = "formatted-pre-2-nisan2019.csv"
f2 = "formatted-pre-2-mayis2019.csv" 
f3 = "formatted-pre-2-haziran2019.csv"

output_f = 'formatted-all-month-alarms.csv' 

PATH = "/home/waris/Github/tupras-analysis/data/"
path = PATH + "/processed/alarms/"

df0 = pd.read_csv(path + f0, parse_dates = ["StartTime","EndTime"])
df1 = pd.read_csv(path + f1, parse_dates = ["StartTime","EndTime"])
df2 = pd.read_csv(path + f2, parse_dates = ["StartTime","EndTime"])
df3 = pd.read_csv(path + f3, parse_dates = ["StartTime","EndTime"])


# %%
df0.tail(2)


# %%
df1.tail(2)


# %%
df2.tail(2)


# %%
df3.tail(2)


# %%
df = pd.concat([df0,df1, df2, df3], ignore_index=True)
df.to_csv( path + output_f, index=False)
df

# %% [markdown]
# # Concatinating Operator Files

# %%
f1 = "operator-pre-1-MarchOperation_v2.csv"
f2 = "operator-pre-1-AprilOperation_v2.csv"
f3= "operator-pre-1-MayOperation_v2.csv"
f4 = "operator-pre-1-JuneOperation_v2.csv"

output_f = 'operator-all-month-actions.csv' 

path = PATH + "/processed/operator-actions/"
df0 = pd.read_csv(path + f1, parse_dates = ["EventTime"])
df1 = pd.read_csv(path + f2, parse_dates = ["EventTime"])
df2 = pd.read_csv(path + f3, parse_dates = ["EventTime"])
df3 = pd.read_csv(path + f4, parse_dates = ["EventTime"])

df = pd.concat([df0,df1, df2, df3], ignore_index=True)
df.to_csv( path + output_f, index=False)
df


# %%


