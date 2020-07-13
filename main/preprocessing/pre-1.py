# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import json
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from pathlib import Path


# %%
def changeDate(d):
        d = d.replace(".000000000","")
        d = d.replace("/","-")
        return parse(d)

def getMessageType(message):
    if message.find("Recover") != -1:
        return "Recover"
    elif message.find("NR") != -1:
        return "NR"
    else:
        return "Activation"

# %%
msg_filter = " ACK" # to filter Ack messages
path = "/home/waris/Github/tupras-analysis/data/"
path_raw = path+"raw/alarms/"  
path_prcessed = path+"processed/alarms/"


#%%
for f in ["haziran2019.csv","march2019.csv","mayis2019.csv","nisan2019.csv"]:    
    
    print("============== File Name:{} =================".format(f))
    input_file_name = f
    output_flie_name = "formatted-pre-1-"+ input_file_name
    print(">> Debug",path_raw+input_file_name)
    df = pd.read_csv(path_raw+input_file_name, delimiter= ";" ,encoding = "ISO-8859-1")    

    print("Column  Type")
    for col in df.columns:
        print(col, type(df[col][0]))
        if isinstance(df[col][0],str):
            df[col] = df[col].apply(lambda s: " ".join(s.split()))

    print(type(df["EventTime"][0]))

    
    df["EventTime"] = df["EventTime"].apply(changeDate)
    df["MessageType"] = df["Message"].apply(getMessageType)
    df2 = df.loc[df['Message'].map(lambda arg: arg.find(msg_filter)) == -1] # df_temp[~df_temp["Message"].str.contains(msg_filter,case=True)] # filtering Ack messages. 
    print(df.shape,df2.shape)
    df = df2
    df.to_csv(path_prcessed+output_flie_name, index=False)
    print(df.shape,df2.shape)
print("========================= Complete =====================")

# %% [markdown]
# # Preprocessing of Operator Data

# %%
files_operator = ["MarchOperation_v2.xls","AprilOperation_v2.xls","MayOperation_v2.xls","JuneOperation_v2.xls"]
path_raw = path+"raw/operator-actions/"
path_prcessed = path+"processed/operator-actions/"
for f in files_operator:
    output_flie_name = "operator-pre-1-"+ f.split(".")[0]+".csv"
    print("==================== File : {} =============".format(f))
    cols = ["MachineName","SourceName","EventTime","Message","Severity","Mask","NewState","EventType","EventCategory","AckReq","ActorID","Area","Attributes"]
    df_excel_operator = pd.read_excel(path_raw+f,usecols=cols)

    for col in df_excel_operator.columns:
        if isinstance(df_excel_operator[col][0],str):
            df_excel_operator[col] = df_excel_operator[col].apply(lambda s: " ".join(s.split()))

    df_excel_operator["EventTime"] = df_excel_operator["EventTime"].apply(changeDate)
    df_excel_operator.to_csv(path_prcessed+output_flie_name, index=False)


# %%


