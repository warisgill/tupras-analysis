

#%%

import time
from helpers import alarms
# import plotly.express as px
import plotly.graph_objects as go



#%%
"""  Lodading the Data and Preprocessing """
PATH = "/home/waris/Github/tupras-analysis/data/"
alarm_file_path = PATH + "processed/alarms/final/final-all-months-alarms.csv"
op_action_file_path = PATH + "processed/operator-actions/final/final-all-month-actions.csv"

start = time.time()

df =alarms.loadAlarmsData(file_path=alarm_file_path)
# df_main_actions = alarms.loadOperatorData(file_path=op_action_file_path)

""" Common Sources in all months. Try it but can be skipped. """
# df_main_alarms = getDFWithCommonSourcesInAllMonths(df_main_alarms)

""" Chaning name 2 alias for alarm data but skipping it """
# source2Alias, alias2source = convertSourceNamesToAlias(df_main_alarms)

print("Total Time to load the data ", time.time()-start)
print(f"Number of Unique Alarm Sources = {len(df['SourceName'].unique())}")

#%%

"""
    Condition Analysi
"""


conditions_count = dict(df["Condition"].value_counts())

trace1 = go.Bar(x=list(conditions_count.keys()),y=list(conditions_count.values()), name="Conditions",text=list(conditions_count.values()), textposition='outside' )

fig = go.Figure()
fig.add_trace(trace1)

fig.update_layout(
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Number of Alarms',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Condition",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    height=600, 
    width=1100,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside',textfont_size=80)
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()






# %%
""" 
    Filter the Alarm Data
    1. Ignore the communication Alarms
    2. Ignore the momentary alarms => 20 seconds
    3. Remove Staling Alarms => 12 hours    
    4. Remove sources which are triggered less 20 in whole dataset
    5. Include all the months
    6. DO SKIP ANY SOURCENAME IF IGNORING COMMUNICATION ALARMS
"""
# ignore_comm_alarms = False
# momentary_alarms_f = 10  # seconds
# staling_alarm_f = (60*60) * 12 # hours
# min_alarms_per_source_f = 20 # any source which is not triggered atleast 20 times in whole dataset will be removed
# months_f = df_main_alarms["Year-Month"].unique()
# snames_f = [] # ONLY USE IF NOT IGNORING COMM ALRMS

# df_alarms_new = alarms.filterAlarmData(df_main_alarms, months=months_f, sources_filter=snames_f,monmentarly_filter=momentary_alarms_f, staling_filter=staling_alarm_f, ingore_communication_alarms=ignore_comm_alarms, min_alarms_per_source=min_alarms_per_source_f)

#%%
"""
    use case=> Bandwidht and Storage Reduction
    Retention period.

    Suppose that on average each alarm takes roughly 1KB of sotorage space.
"""
month2_A_count = dict(df["Year-Month"].value_counts())
month2_S_A_count_10_sec=  dict(df.loc[df["TimeDelta"]>=10]["Year-Month"].value_counts())
month2_S_A_count_20_sec=  dict(df.loc[df["TimeDelta"]>=20]["Year-Month"].value_counts())



x_sorted = alarms.sortMonthYearTuple(month2_A_count.keys()) 

print("hello" ,x_sorted)
y1 = [month2_A_count[month_year] for month_year in x_sorted]
trace1 = go.Bar(x=x_sorted,y=y1, name="Raw Alarms",text=y1, textposition='outside' )
y2 = [month2_S_A_count_10_sec[month_year] for month_year in x_sorted]
trace2 = go.Bar(x= x_sorted, y= y2,name="10s Filter",text=y2, textposition='outside')
y3 = [month2_S_A_count_20_sec[month_year] for month_year in x_sorted]
trace3 = go.Bar(x= x_sorted, y= y3,name="20s Filter",text=y3, textposition='outside')

fig = go.Figure()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig.add_trace(trace3)

fig.update_layout(
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='Number of Alarms',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Month",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    height=600, 
    width=1100,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.update_traces(texttemplate='%{text:.3s}', textposition='outside',textfont_size=80)
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()


# %%

x_sorted = alarms.sortMonthYearTuple(month2_A_count.keys()) 


# y1 = [month2_A_count[month_year] for month_year in x_sorted]
# trace1 = go.Bar(x=x_sorted,y=y1, name="",text=y1, textposition='outside' )
y2 = [(month2_S_A_count_10_sec[month_year]/month2_A_count[month_year])*100 for month_year in x_sorted]
trace2 = go.Bar(x= x_sorted, y= y2,name="10s Filter",text=y2, textposition='outside')
# y3 = [(month2_S_A_count_20_sec[month_year]/month2_A_count[month_year])*100 for month_year in x_sorted]
# trace3 = go.Bar(x= x_sorted, y= y3,name="20s Filter",text=y3, textposition='outside')

fig = go.Figure()
# fig.add_trace(trace1)
fig.add_trace(trace2)
# fig.add_trace(trace3)

fig.update_layout(
    xaxis_tickfont_size=14,
    yaxis=dict(
        title='% of reduction in Storage Space',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Month",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    barmode='group',
    height=600, 
    width=1100,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
)

fig.update_traces(texttemplate='%{text:.3s}%', textposition='outside',textfont_size=80)
# fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
fig.show()

# %%


print(f"Average of Storage reduction: {sum(y2)/len(y2)}")
# %%





