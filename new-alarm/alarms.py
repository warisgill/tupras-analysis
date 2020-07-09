import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import plotly.io as pio
from dateutil.parser import parse
from plotly.subplots import make_subplots

# pio.orca.config.use_xvfb = True
pio.orca.config.executable = "/home/waris/anaconda3/envs/tupras/bin/orca"


def frequencyOfAlarmsActivated(alarms, timediff=60):
    alarms_by_start_time = [alarm for alarm in sorted(
        alarms, key=lambda arg: arg["StartTime"], reverse=False)]
    alarms_by_end_time = [alarm for alarm in sorted(
        alarms, key=lambda arg: arg["EndTime"], reverse=False)]
    freq = []
    max_delta = -1
    temp = 0
    for i in range(len(alarms)):
        t_end = alarms_by_end_time[i]["EndTime"]
        j = 0
        for j in range(temp, len(alarms)):
            t_start = alarms_by_start_time[j]["StartTime"]
            delta = timedelta.total_seconds(t_start - t_end)

            if delta < 0:
                #                 print(delta)
                continue
            else:
                temp = j-1
                if max_delta < delta:
                    max_delta = delta
#                 print(delta)
                freq.append(delta)
                break
#     print("Max seconds :", max_delta)
    return freq


def maskSourceNames(df):
    change_source_names = {}
    index = 1
    for sname in df["SourceName"].unique():
        change_source_names[sname] = "S{}".format(index)
        index += 1

    df["SourceName"] = df["SourceName"].apply(
        lambda sname: change_source_names[sname])

    return change_source_names


def getMessageType(message):

    if message.find("Recover") != -1:
        return "Recover"
    elif message.find("NR") != -1:
        return "NR"
    else:
        return "Activation"

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


def findChatterings(alarms, chattering_timedelta_threshold=60.0, chattering_count_threshold=3):
    """Find the chatterings in an alarms list from the same source. 

    Parameters
    ----------
    alarms  : list of dict
        A list of alarms from the same source. 
    chattering_timedelta_threshold : float, optional
        Duration in seconds for which to finde cattering alarms, by default 60.0 seconds.
    chattering_count_threshold : int, optional
        Threshold for minimum number of alarms to be activated in duration of chattering_timedelta_threshold, by default 3

    Returns
    ----------
    chattering : dict
        It contains the StartTime as key of chattering, and a dict as a value which is 
        consits of an index and a count of of alarms chattered within chattering_timedelta_threshold next 
        to it.  
    """

    chattering = {}
    alarms = [alarm for alarm in sorted(alarms, key=lambda arg: arg["StartTime"], reverse=False)]
    i = 0
    j = 0

    while i < (len(alarms)):
        prev_start = alarms[i]["StartTime"]
        prev_end = alarms[i]["EndTime"]
        count_alarms = 0
        j = i + 1
        while j < len(alarms):
            next_start = alarms[j]["StartTime"]
            next_end = alarms[j]["EndTime"]

            # this assert is very important: the prev alarm has to turn off before the start of
            # the next one
            assert(prev_start <= next_start)
            assert(prev_end <= next_start)
            assert(prev_end <= next_end)

            delta = timedelta.total_seconds(next_start - prev_start)
            assert (delta >= 0)
            if delta > chattering_timedelta_threshold:
                break
            count_alarms += 1
#             print(time_delta, "count ++ ", count, t_prev, t_next)
            j += 1
        i = j
        if count_alarms >= chattering_count_threshold:
            chattering[prev_start] = {"index": i, "count": count_alarms}

    return chattering


def getChatteringAlarmsFromDataFrame(df):
    """It uses convertRecordsToAlarms function to convert records of DF to alarms.

    Parameters
    ----------
    df : pandas dataframe


    Returns
    -------
    chatters: list of tuple
        Each tuple (SourceName,,) in the list is an alarm with the StartTime and EndTime of an alarm, found in whole df.
    total_alarms_in_all_chatters: int
        Total number of alarms in chatterings. 
    """

    chatters = []
    total_alarms_in_all_chatters = 0
    for sname in df["SourceName"].unique():
        df_source = df.loc[df['SourceName'].isin([sname])]
        v1 = sname
        v2 = 0
        v3 = 0
        for condition in df_source["Condition"].unique():
            df_condition = df_source.loc[df_source['Condition'].isin([
                                                                     condition])]
            chats = findChatterings(df_condition.to_dict(orient="records"))
            v2 += len(chats.keys())  # number of times it chatters
            # total alarms in one chatter
            v3 += sum([d["count"] for d in chats.values()])
        chatters.append((v1, v2, v3))
        total_alarms_in_all_chatters += v3
    return chatters, total_alarms_in_all_chatters
