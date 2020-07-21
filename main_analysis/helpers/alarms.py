from datetime import timedelta
from pandas import read_csv

# %%

def loadAlarmsData(path, filename, preprocess=True):
    df = read_csv(path + filename, low_memory=False,
                  parse_dates=["StartTime", "EndTime"])

    if preprocess == True:
        addTimeDeltaToDataFrame(df)
        addMonthToDataFrame(df, date_col="StartTime")

    return df

def loadOperatorData(path, filename, preprocess=True):
    df = read_csv(path + filename, low_memory=False, parse_dates=["EventTime"])

    if preprocess == True:
        addMonthToDataFrame(df, date_col="EventTime")

    return df

def filterAlarmData(df, months=None, sources_filter=[], monmentarly_filter=20, staling_filter=(60*60)*24, ingore_communication_alarms=False, min_alarms_per_source=10):

    print(
        f">>Preprocessing... \n   Months to include={months}\n  Ignore Sources={sources_filter}\n  Ingnore Momentarlily Alarms Filter={monmentarly_filter}seconds \n   Ignoreing Staling Alarms Filter={staling_filter/3600.0} hours, \n Ignore Communication Alarms = {ingore_communication_alarms} \n Remove sources whose count is less than {min_alarms_per_source}")
    
    if months is None:
        months = df["Month"].unique()

    df_new = df[(df["TimeDelta"] > monmentarly_filter) & (df["TimeDelta"] < staling_filter) & (
        df["Month"].isin(months)) & (~df["SourceName"].isin(sources_filter))]

    if ingore_communication_alarms==True:
        df_new = df_new[~df_new["Condition"].isin(["IOP", "IOP-"])]

    source2count = dict(df_new["SourceName"].value_counts())
    select_sources = [k for k, v in source2count.items() if v >= min_alarms_per_source]
    df_new = df_new[df_new["SourceName"].isin(select_sources)]

    return df_new

def getDFWithCommonSourcesInAllMonths(df):
    each_month_source_names = [[sname for sname in df[df["Month"]==month]["SourceName"].unique()] for month in df["Month"].unique()]

    common_sourcenames_in_all_months = set.intersection(*[set(l) for l in each_month_source_names])

    df = df[df["SourceName"].isin(common_sourcenames_in_all_months)]
    
    return df

def _getTimeDelta(start_time, end_time):
    delta = timedelta.total_seconds(end_time-start_time)
    assert delta >= 0
    return delta

def _concatenateSourceNameAndCondition(sname, condition):
    return sname+"-"+condition

def addTimeDeltaToDataFrame(df):
    df["TimeDelta"] = df[["StartTime", "EndTime"]].apply(
        lambda arg: _getTimeDelta(*arg), axis=1)

def addMonthToDataFrame(df, date_col):
    df["Month"] = df[date_col].apply(lambda arg: arg.month)

def updatSourceNamewithCondition(df):
    df["SourceName"] = df[["SourceName", "Condition"]].apply(
        lambda arg: _concatenateSourceNameAndCondition(*arg), axis=1)

def convertSourceNamesToAlias(df):
    alias2name = {f"A{k}": v for k, v in enumerate(df["SourceName"].unique())}
    name2alias = {v: k for k, v in alias2name.items()}
    df["SourceName"] = df["SourceName"].apply(lambda sname: name2alias[sname])
    return name2alias, alias2name
    