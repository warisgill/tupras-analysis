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