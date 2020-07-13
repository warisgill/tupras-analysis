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


def findChatteringsv2(alarms, chattering_timedelta_threshold=60.0, chattering_count_threshold=3):
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
    alarms_without_chatters = []
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
            if prev_end > next_start:
                # print(alarms[i]["SourceName"], alarms[j]["SourceName"])
                print(f"p_start = {prev_start} p_end:{prev_end} next_start={next_start} next_end ={next_end}")
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

        alarms_without_chatters.append(alarms[i])
        if count_alarms >= chattering_count_threshold:
            chattering[prev_start] = {"index": i, "count": count_alarms}
            i = j
        else:
            i += 1
        

    return chattering, alarms_without_chatters


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
