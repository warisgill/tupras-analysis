from datetime import timedelta
import plotly.graph_objs as go


# ================= Section 1: ================
def removeShortDurationAlarms(df, duration_filters=[20,120]):
    df_f1 = df[df["TimeDelta"]>duration_filters[0]] # keep the alarms whose duration is larger than the filter
    df_f2 = df[df["TimeDelta"]>duration_filters[1]]
    
    
    d = df["Month"].value_counts()
    mcounts = [t for t in sorted(d.items(), key= lambda arg: arg[0])]
    trace1 = go.Bar(name='Without filter',x=[t[0] for t in mcounts], y= [t[1] for t in mcounts ],  text=[t[1] for t in mcounts],  textposition='auto')

    
    d = df_f1["Month"].value_counts()
    mcounts = [t for t in sorted(d.items(), key= lambda arg: arg[0])]
    trace2 = go.Bar(name='{}s filter'.format(duration_filters[0]),x=[t[0] for t in mcounts], y= [t[1] for t in mcounts ],  text=[t[1] for t in mcounts],  textposition='auto')
    
    d = df_f2["Month"].value_counts()
    mcounts = [t for t in sorted(d.items(), key= lambda arg: arg[0])]
    trace3 = go.Bar(name="{}s filter".format(duration_filters[1]),x=[t[0] for t in mcounts], y= [t[1] for t in mcounts ],  text=[t[1] for t in mcounts],  textposition='auto')
        
    
    fig = go.Figure()
    fig.add_trace(trace1)
    fig.add_trace(trace2)
    fig.add_trace(trace3)
    fig.update_layout(barmode='group', yaxis=dict(
        title='Count',
        titlefont_size=16,
        tickfont_size=14,
    ),xaxis=dict(
        title='Month',
        titlefont_size=16,
        tickfont_size=14,
    ))
    fig.show()
    

def storageAnalysis(df,duration_filters=[20,120]):
    df_f1 = df[df["TimeDelta"]>duration_filters[0]] # keep the alarms whose duration is larger than the filter
    df_f2 = df[df["TimeDelta"]>duration_filters[1]]
    
    x_axis = ["Raw Alarms", "After {}s filter".format(duration_filters[0]),"After {}s filter".format(duration_filters[1])]
    y_axis = [df.shape[0], df_f1.shape[0],df_f2.shape[0]]
    
    trace = go.Bar(x=x_axis, y=y_axis, text=y_axis,  textposition='auto')
        
    
    fig = go.Figure()
    fig.add_trace(trace)
    fig.update_layout(yaxis=dict(
        title='Storage Utilization (KB)',
        titlefont_size=16,
        tickfont_size=14,
    ))
    fig.show()
