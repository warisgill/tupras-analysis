import math
import numpy as np
import pandas as pd
from datetime import timedelta
from datetime import datetime
from dateutil.parser import parse
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

def findChatterings(alarms, chattering_timedelta_threshold=60.0, chattering_count_threshold=3):
    chattering = {}
    
    alarms = [alarm for alarm in sorted(alarms, key=lambda arg: arg["StartTime"], reverse=False)]    
    count = 0
    
    for i in range(len(alarms)):
        t_prev = alarms[i]["StartTime"]
        count = 0
        for j in range(i+1, len(alarms)):
            t_next = alarms[j]["StartTime"]    
            
            if timedelta.total_seconds(t_next - t_prev) > chattering_timedelta_threshold:
                break
                
            count += 1
#             print(time_delta, "count ++ ", count, t_prev, t_next)
        
#         if count > chattering_count_threshold:
        chattering[t_prev] = {"own_index": i, "count": count}
            
    return chattering


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