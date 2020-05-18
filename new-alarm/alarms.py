import math
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse
import plotly.io as pio
pio.orca.config.use_xvfb = True
pio.orca.config.executable = "/usr/local/bin/orca"
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def frequencyOfAlarmsActivated(alarms, timediff=60):
    alarms_by_start_time = [alarm for alarm in sorted(alarms, key=lambda arg: arg["StartTime"], reverse=False)]
    alarms_by_end_time = [alarm for alarm in sorted(alarms, key=lambda arg: arg["EndTime"], reverse=False)]
    freq = []
    max_delta = -1
    temp = 0
    for i in range(len(alarms)):
        t_end = alarms_by_end_time[i]["EndTime"]
        j = 0 
        for j in range(temp,len(alarms)):        
            t_start = alarms_by_start_time[j]["StartTime"]
            delta = timedelta.total_seconds(t_start - t_end)
            
            if delta< 0:
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
    
    df["SourceName"] = df["SourceName"].apply(lambda sname: change_source_names[sname])
    
    return change_source_names

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
        t_prev = alarms[i]["StartTime"]
        count_alarms = 0
        j = i + 1 
        while j < len(alarms):
            t_next = alarms[j]["StartTime"]    
            if timedelta.total_seconds(t_next - t_prev) > chattering_timedelta_threshold:
                break
            count_alarms += 1
#             print(time_delta, "count ++ ", count, t_prev, t_next)
            j +=1
        i = j
        if count_alarms >= chattering_count_threshold:
            chattering[t_prev] = {"index": i, "count": count_alarms}

    return chattering


def convertRecordsToAlarms(records):
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


    alarms = [] # conainsts alarms with start and end time. 
    conditions_queues = {} # for enqueue and deque of records., Needed dictionary because there can be multiple types of alarms from the same source. 
    alarm = None # dictionary 
    records = [v for v in sorted(records, key=lambda arg: arg["EventTime"], reverse = False)]
    for record in records:
        
        if conditions_queues.get(record["Condition"]) == None: # initiazlize the queue 
            conditions_queues[record["Condition"]] = []

        
        if record["Message"].find("Recover") == -1 and record["Message"].find("NR") == -1: # Enqueue the record
            conditions_queues[record["Condition"]].append(record) 
        else: 
            if len(conditions_queues[record["Condition"]])== 0:
                continue

            alarm = conditions_queues[record["Condition"]].pop(0) # Dqueue the record
            alarm = {k:v for k,v in alarm.items()}
            alarm["StartTime"] = alarm["EventTime"]
            alarm["EndTime"] = record["EventTime"]
            alarm["EndMessage"] = record["Message"]
            del alarm["EventTime"]
            alarms.append(alarm)

    return alarms