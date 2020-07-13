
#%%
from datetime import timedelta
import pandas as pd
from helper_storage import removeShortDurationAlarms
from helper_storage import storageAnalysis

filter_short_alarms = [20, 40]  #


#%%
PATH = "/home/waris/Github/tupras-analysis/data/"
path = PATH + "/processed/alarms/"
alarms_fname = "formatted-all-month-alarms.csv" 
df_main_alarms = pd.read_csv(path + alarms_fname, low_memory=False ,parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(lambda arg: timedelta.total_seconds(arg)) 
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(lambda arg: arg.month)


#%%
"""
    use case=> Bandwidht and Storage Reduction
    Retention period.

    Suppose that on average each alarm takes roughly 1KB of sotorage space.
"""


removeShortDurationAlarms(df_main_alarms)
storageAnalysis(df_main_alarms)

print(">> Done")
