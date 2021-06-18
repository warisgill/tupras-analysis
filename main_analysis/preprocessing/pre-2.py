
"""
    Summary
    Record mean 1 row of the csv file. 
    # This notebook converts the records into alarms. 
"""


# %%
import pandas as pd
import glob

#%%

def maskSourceNames(df):
    change_source_names = {}
    index = 1
    for sname in df["SourceName"].unique():
        change_source_names[sname] = "S{}".format(index)
        index += 1

    df["SourceName"] = df["SourceName"].apply(
        lambda sname: change_source_names[sname])

    return change_source_names


def convertRecordsToAlarmsV1(df_source):
    alarms = []
    for condition in df_source["Condition"].unique():
        df_condition = df_source.loc[df_source['Condition'].isin([condition])]
        df_start = df_condition.loc[df_condition['MessageType'].isin([
                                                                     "Activation"])]
        end_types = [t for t in df_condition["MessageType"].unique() if t !=
                 "Activation"]
        # print(types)
        df_end = df_condition.loc[df_condition['MessageType'].isin(end_types)]
        alarms += getAlarmsFromDFs(df_start, df_end)
    return alarms


def getAlarmsFromDFs(df_start, df_end):
    alarms = []
    start_records = [v for v in sorted(df_start.to_dict(
        orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    end_records = [v for v in sorted(df_end.to_dict(
        orient="records"), key=lambda arg: arg["EventTime"], reverse=False)]
    i = 0
    j = 0
    # print("End len",len(end_records), "Start len", len(start_records))
    while j < len(end_records):
        # print(i,j)
        if len(start_records)>0 and end_records[j]["EventTime"] < start_records[i]["EventTime"]:
            j += 1
        else:
            break

    while i < len(start_records):
        
        if j <len(end_records) and start_records[i]["EventTime"] <= end_records[j]["EventTime"]:
            if i+1 < len(start_records) and start_records[i+1]["EventTime"] < end_records[j]["EventTime"]: # check for the next record
                i += 1
                continue
            alarm = {k: v for k, v in start_records[i].items()}
            alarm["StartTime"] = alarm["EventTime"]
            alarm["EndTime"] = end_records[j]["EventTime"]
            alarm["EndMessage"] = end_records[j]["Message"]
            del alarm["EventTime"]
            alarms.append(alarm)
            j += 1
        elif j <len(end_records) and start_records[i]["EventTime"] > end_records[j]["EventTime"]:
            j +=1
            continue   
             
        i += 1

    return alarms

def _convertRecordsToAlarmsOld(records):
    """ Convert records from the same source to proper alarms with start and end time.   

        The record which contains "Recover" or "NR" in the Message column shows the deactivations. 

    Parameters
    ----------
    records : list of dict
        Each dict represent either activation of an alarm or deactivation of an alarm.  

    Returns
    -------
    alarms : list of dict
        Each dict in the list is an alarm with the StartTime and EndTime of an alarm. 
    """
    alarms = []  # conainsts alarms with start and end time.
    # for enqueue and deque of records., Needed dictionary because there can be multiple types of alarms from the same source.
    conditions_queues = {}
    alarm = None  # dictionary
    records = [v for v in sorted(
        records, key=lambda arg: arg["EventTime"], reverse=False)]
    for record in records:

        # initiazlize the queue
        if conditions_queues.get(record["Condition"]) == None:
            conditions_queues[record["Condition"]] = []

        # Enqueue the record
        if record["Message"].find("Recover") == -1 and record["Message"].find("NR") == -1:
            conditions_queues[record["Condition"]].append(record)
        else:
            if len(conditions_queues[record["Condition"]]) == 0:
                continue

            alarm = conditions_queues[record["Condition"]].pop(
                0)  # Dqueue the record
            alarm = {k: v for k, v in alarm.items()}
            alarm["StartTime"] = alarm["EventTime"]
            alarm["EndTime"] = record["EventTime"]
            alarm["EndMessage"] = record["Message"]
            del alarm["EventTime"]
            alarms.append(alarm)

    return alarms



# %%

cols = ["MachineName","SourceName","EventTime", "Message","MessageType","Condition"]
PATH = "../../data"
path = PATH + "/processed/alarms_with_feed/"

# %%
for p in glob.glob(path+"*.csv"):  
    print(f"==================== File : {p.split('/')[-1]} =============")
    
    df = pd.read_csv(p, low_memory=False, usecols=cols,parse_dates=["EventTime"])

    assert len(df["MachineName"].unique()) == 1 # all the alarms should be related to the same unit
    
    alarms = []
    differs = []
    sources_ranks_dict = df['SourceName'].value_counts()
    id = 0 # for debugging
    for sname in sources_ranks_dict.keys():
        id += 1
        df_sname = df.loc[df['SourceName'].isin([sname])] # source DF
        types_rank_dict = df_sname["MessageType"].value_counts() # source ranks
        total = 0
        for key in types_rank_dict.keys():
            total += types_rank_dict[key]
        assert(total== sources_ranks_dict[sname]) # sum is equal to count 
        
        source_alarms = convertRecordsToAlarmsV1(df_sname)
        alarms += source_alarms

        temp_alarms2 = _convertRecordsToAlarmsOld(df_sname.to_dict(orient="records"))
        if len(source_alarms)-len(temp_alarms2) != 0:
            print(">>[{}]Source: {},Conditions:{}".format(id,sname, df_sname["Condition"].unique()), end="=>")
            print("ALARMS1:{},Alarms2:{},Diff(new-old):{}".format(len(source_alarms),len(temp_alarms2), len(source_alarms)-len(temp_alarms2)),end="")
            print("")

       
        if (len(source_alarms) != len(temp_alarms2)):
            differs.append(sname)

    # writing to files
    df_out = pd.DataFrame(alarms) 
    out_fname = f"pre-2-alarms-{p.split('/')[-1].split('-')[-1]}"  
    df_out.to_csv(path+out_fname, index = False)
    print("Differs in 2 Algos",differs,len(differs))
    print(">> ",df_out.info())


print(">> Complete")

# ==== All Below Cells are for debugging ==========================

# %% [markdown]
# # Section Debugging

# %%



# %%
# print(differs,len(differs))
# sname = "47PI1734"
# df_source = df.loc[df['SourceName'].isin([sname])]

# alarms1 = convertRecordsToAlarmsV1(df_source)
# alarms2 = convertRecordsToAlarmsV2(df_source.to_dict(orient="records"))
# len(alarms1), len(alarms2)


# %%
# s= "47PI1734"
# df3 = pd.read_csv(path + files[0], low_memory=False, usecols=cols ,parse_dates=["EventTime"]) # may file
# temp_df=df3.loc[df3["SourceName"].isin([s])]

# alarms1 = convertRecordsToAlarmsV1(temp_df)
# alarms2 = _convertRecordsToAlarmsOld(temp_df.to_dict(orient="records"))

# temp_df2 = temp_df.sort_values(by=["EventTime"])

# temp_df2.to_csv(path+"debug.csv")
# print(len(alarms1), len(alarms2))

# temp_df2


# %%
# # for index, row in temp_df2.iterrows():
# #     print(row['EventTime'], row['Condition'], row["MessageType"])
# temp_df2


# %%
# print(s)
# df4 = pd.read_csv(path + "formatted-pre-2-mayis2019.csv", low_memory=False)
# df4.loc[df4["SourceName"].isin([s])]


# %%
# findChatteringsv2(alarms1)

