# %%
"""
    Concatinating dfs i.e., all alarms into 1 file
"""


# %%
import glob
import pandas as pd
from datetime import datetime, timedelta

output_f = 'final-all-months-alarms.csv' 
PATH = "/home/waris/Github/tupras-analysis/data/"
path = PATH + "/processed/alarms/"

# %%



fps = [f for f in glob.glob(path+ "*.csv") if f.find("pre-2-alarms-")!=-1]
print(f">> Files to process {fps}")
dfs_list = []
for f in fps:
    print(f">> === File: {f.split('/')[-1]}")
    df = pd.read_csv(f, parse_dates = ["StartTime","EndTime"])
    
    df["TimeDelta"] = df[["StartTime", "EndTime"]].apply(lambda arg: timedelta.total_seconds(arg[1]-arg[0]) , axis=1)
    df["Year-Month"] =df["StartTime"].apply(lambda arg: (arg.year,arg.month))
    print(f">>Uninque month and year {df['Year-Month'].unique()}")
    dfs_list.append(df)

df = pd.concat(dfs_list, ignore_index=True)
df.to_csv( path+"final/"+ output_f, index=False)
df


# %%
"""
    Concatinating operator files
"""
# ServerProgID ServerNodeName	SubscriptionName	SourceName	EventTime	EventTime_MS	Severity	Message	Quality	Condition	SubCondition	Mask	NewState	EventType	EventCategory	AckReq	ActiveTime	ActiveTime_MS	Cookie	ActorID	Attributes	Area


path = PATH + "/processed/operator-actions/"

cols = ["MachineName","SourceName","EventTime" ]
op_dfs = []
for f in glob.glob(path+"*.csv"):
    print(f">> Processing Op {f.split('/')[-1]}")
    df= pd.read_csv(f, parse_dates=["EventTime"], usecols=cols)
    df["Year-Month"] = df["EventTime"].apply(lambda arg: (arg.year,arg.month))
    print(">> unique Year and Month",df["Year-Month"].unique())
    op_dfs.append(df)



df = pd.concat(op_dfs, ignore_index=True)
df.to_csv( path+"final/final-all-month-actions.csv", index=False)
df





# %%

# # testing
# path = PATH + "/processed/alarms/"
# df = pd.read_csv(path+"pre-2-alarms-2019_6.csv", parse_dates=["StartTime"])
# df["Year-Month"] = df["StartTime"].apply(lambda arg: (arg.year,arg.month))


# %%
# df["Year-Month"].unique()

# df[df["Year-Month"]==(2019,6)]
# %%
