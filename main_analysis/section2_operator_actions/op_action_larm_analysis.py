# %%

from datetime import timedelta
import networkx as nx
import pandas as pd
import plotly.express as px

from section2_operator_actions.helper_operator_actions import filterEdges, getSourceNamesWhichRequiredActionAndNot, getTrueAndNuisanceSourceNames

from section2_operator_actions.helper_operator_actions import (getFinalOperatorAlarmRelationGraph,
                                     plotOperatorAlarmRelationHeatMap,
                                     preProcessAlarmData)

# %%

PATH = "/home/waris/Github/tupras-analysis/data/"
path_alarms = PATH + "processed/alarms/"
path_op_actions = PATH + "processed/operator-actions/"

# %%

alarms_fname = "formatted-all-month-alarms.csv"
operator_fname = "operator-all-month-actions.csv"
df_main_alarms = pd.read_csv(path_alarms + alarms_fname,
                             low_memory=False, parse_dates=["StartTime", "EndTime"])
df_main_alarms["TimeDelta"] = df_main_alarms["EndTime"] - \
    df_main_alarms["StartTime"]
df_main_alarms["TimeDelta"] = df_main_alarms["TimeDelta"].apply(
    lambda arg: timedelta.total_seconds(arg))
df_main_alarms["Month"] = df_main_alarms["StartTime"].apply(
    lambda arg: arg.month)

df_main_actions = pd.read_csv(
    path_op_actions + operator_fname, low_memory=False, parse_dates=["EventTime"])
df_main_actions["Month"] = df_main_actions["EventTime"].apply(
    lambda arg: arg.month)


# %%
df_main_actions.info(), df_main_alarms.info()

# %%
""" 
    Common Sources in all months. Try it but can be skipped. 
"""
# each_month_source_names = [[sname for sname in df_main_alarms[df_main_alarms["Month"]==month]["SourceName"].unique()] for month in df_main_alarms["Month"].unique()]

# common_sourcenames_in_all_months = set.intersection(*[set(l) for l in each_month_source_names])

# df_main_alarms = df_main_alarms[df_main_alarms["SourceName"].isin(common_sourcenames_in_all_months)]

# df_main_alarms.info()


# %%
filter_short_alarms = [20, 120]  # seconds
staling_alarms_filter = (60*60) * 12

months_f = df_main_alarms["Month"].unique()
num_sub_graphs = 8
min_intersection_f = num_sub_graphs//2 + 1
snames_f = ["47TI1713"]
edge_filter = 8
# put month filter on operator data and then pass
df_alarms_new = preProcessAlarmData(df_main_alarms, months=months_f,
                                    sources_filter=snames_f, monmentarly_filter=20, staling_filter=60*60*12)
df_actions_new = df_main_actions[df_main_actions["Month"].isin(months_f)]

temp_g = getFinalOperatorAlarmRelationGraph(df_alarms=df_alarms_new, df_actions=df_actions_new,
                                            num_graphs=num_sub_graphs, min_graphs_intersection_filter=min_intersection_f)

main_g = filterEdges(temp_g, edge_drop_factor=edge_filter)
print(">> Done")

# %%
print(">> # of Edges Main graph1 {} and in graph2 {}".format(
    main_g.number_of_edges(), main_g.number_of_edges()))
operator_nodes = [
    action for action in main_g.nodes if action.find("Operator") != -1]

print(f">> Total number of operator nodes in the graph={len(operator_nodes)}")
print(
    f">> Total number of Alarm Tags in the graph = {len(main_g.nodes)- len(operator_nodes)}")


# %%
"""
    Plotting actions and without action alarms
"""

sources_need_action, sources_dont_need_action = getSourceNamesWhichRequiredActionAndNot(
    df_alarms_new, main_g)

total_number_of_alarms = df_alarms_new.shape[0]
number_of_alarms_not_have_action = df_alarms_new[df_alarms_new["SourceName"].isin(
    sources_dont_need_action)].shape[0]

x_axis = ["Raw Alarms", "True Alarms", "Nuisance Alarms"]
y_axis = [total_number_of_alarms, total_number_of_alarms -
          number_of_alarms_not_have_action, number_of_alarms_not_have_action]

fig = px.bar(x=x_axis, y=y_axis)
fig.update_layout(yaxis=dict(title="# number of alarms",
    titlefont_size=16,
    tickfont_size=14,
), xaxis=dict(
    title='',
    titlefont_size=16,
    tickfont_size=14,
))
fig.show()

x_axis = ["True Alarms", "Nuisance Alarms"]
y_axis = [((total_number_of_alarms-number_of_alarms_not_have_action)/total_number_of_alarms)
          * 100, (number_of_alarms_not_have_action/total_number_of_alarms)*100]

fig = px.bar(x=x_axis, y=y_axis)
fig.update_layout(yaxis=dict(title="% of total alarms",
    titlefont_size=16,
    tickfont_size=14,
), xaxis=dict(
    title='',
    titlefont_size=16,
    tickfont_size=14,
))
fig.show()


# %%

assert df_alarms_new.shape[0] == (df_alarms_new[df_alarms_new["SourceName"].isin(
    sources_need_action)].shape[0] + df_alarms_new[df_alarms_new["SourceName"].isin(sources_dont_need_action)].shape[0])

print(f"\n>>Total number of operator Actions = {df_actions_new.shape[0]}")
print(
    f"\n\n>> Source which required operator action: Total = {len(sources_need_action)},  Sources = {sources_need_action}")
print(
    f"\n\n>> Sources dont need action: Total = {len(sources_dont_need_action)}, {sources_dont_need_action}")


# %%

true_sources, nuisance_sources = getTrueAndNuisanceSourceNames(df_alarms_new,df_actions_new)
print(true_sources, nuisance_sources)