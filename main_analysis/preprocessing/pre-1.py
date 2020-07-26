# %%
import json
import pandas as pd
from datetime import datetime
from dateutil.parser import parse
from pathlib import Path
import glob


# %%
msg_filter = " ACK" # to filter Ack messages
path = "/home/waris/Github/tupras-analysis/data/"
path_raw = path+"raw/alarms/"  
path_prcessed = path+"processed/alarms/"

def getMessageType(message):
    if message.find("Recover") != -1:
        return "Recover"
    elif message.find("NR") != -1:
        return "NR"
    else:
        return "Activation"

#%%

for p in glob.glob(path_raw+"*.csv"):        
        
    print("============== File Name:{} =================".format(p.split("/")[-1]))
    
    df = pd.read_csv(p, delimiter= ";" ,encoding = "ISO-8859-1")
    df.columns = [''.join(e for e in col if e.isalnum()) for col in df.columns]
    df.columns = [col.replace("ï","") for col in df.columns]    
    
    print("Column  Type")
    for col in df.columns:
        print(col, type(df[col][0]))
        if isinstance(df[col][0],str):
            try:
                df[col] = df[col].apply(lambda s:" ".join(s.split()) )
            except:
                print(">> Error =============== String spliting opertion is not applied on ", col)

    print(type(df["EventTime"][0]))
    
    df["EventTime"] = df["EventTime"].apply(lambda d: d.replace(".000000000","").replace("/","-"))
    df["MessageType"] = df["Message"].apply(getMessageType)
    df2 = df.loc[df['Message'].map(lambda arg: arg.find(msg_filter)) == -1]  
    print(f">> Difference after filtering the acks: Before {df.shape} After={df2.shape}")
    df = df2
    output_flie_name = "f-pre-1-"+ p.split("/")[-1]
    df.to_csv(path_prcessed+output_flie_name, index=False)
    del df
    del df2


print("\n\n>> Reading newer receive files.")

for p in glob.glob(path_raw+"new/*.csv"):    
    print("============== File Name:{} =================".format(p.split("/")[-1]))
    

    df = pd.read_csv(p, delimiter= "," ,encoding = "ISO-8859-1")
    df.columns = [''.join(e for e in col if e.isalnum()) for col in df.columns]
    df.columns = [col.replace("ï","") for col in df.columns]    

    print("Column  Type")
    for col in df.columns:
        print(col, type(df[col][0]))
        if isinstance(df[col][0],str):
            df[col] = df[col].apply(lambda s: " ".join(s.split()))

    print(type(df["EventTime"][0]))

    df["EventTime"] = df["EventTime"].apply(lambda d: d.replace(".000000000","").replace("/","-"))
    df["MessageType"] = df["Message"].apply(getMessageType)
    df2 = df.loc[df['Message'].map(lambda arg: arg.find(msg_filter)) == -1] # df_temp[~df_temp["Message"].str.contains(msg_filter,case=True)] # filtering Ack messages. 
    print(f">> Difference after filtering the acks: Before {df.shape} After={df2.shape}")
    df = df2
    output_flie_name = "f-pre-1-"+ p.split("/")[-1]
    df.to_csv(path_prcessed+output_flie_name, index=False)
    del df2
    del df
    
print("========================= Complete Alarms Processing =====================")



# %%
# df = pd.read_csv(path_raw+"new/2020_01.csv", delimiter= "," ,encoding = "ISO-8859-1")
# df
# %%

path_raw = path+"raw/operator-actions/"
path_prcessed = path+"processed/operator-actions/"
for p in glob.glob(path_raw+"*.xls"):
    output_flie_name = "op-pre-1-"+ p.split("/")[-1]
    output_flie_name = output_flie_name.split(".")[0]+".csv" 
    print("==================== File : {} =============".format(p.split("/")[-1]))
    df_excel_operator = pd.read_excel(p)
    df_excel_operator.columns = [''.join(e for e in col if e.isalnum()) for col in df_excel_operator.columns]
    df_excel_operator.columns = [col.replace("ï","") for col in df_excel_operator.columns]

    for col in df_excel_operator.columns:
        print(col, type(df_excel_operator[col][0]))
        if isinstance(df_excel_operator[col][0],str):
            df_excel_operator[col] = df_excel_operator[col].apply(lambda s: " ".join(s.split()))

    df_excel_operator["EventTime"] = df_excel_operator["EventTime"].apply(lambda d: d.replace(".000000000","").replace("/","-"))
    df_excel_operator.to_csv(path_prcessed+output_flie_name, index=False)
    del df_excel_operator


print(">> =========== Processing Op new data")

path_raw = path+"raw/operator-actions/"
path_prcessed = path+"processed/operator-actions/"
for p in glob.glob(path_raw+"new/*.xls"):
    output_flie_name = "op-pre-1-"+ p.split("/")[-1]
    output_flie_name = output_flie_name.split(".")[0]+".csv"

    print("==================== File : {} =============".format(p.split("/")[-1]))
    df_excel_operator = pd.read_excel(p)
    df_excel_operator.columns = [''.join(e for e in col if e.isalnum()) for col in df_excel_operator.columns]
    df_excel_operator.columns = [col.replace("ï","") for col in df_excel_operator.columns]

    for col in df_excel_operator.columns:
        print(col, type(df_excel_operator[col][0]))
        if isinstance(df_excel_operator[col][0],str):
            df_excel_operator[col] = df_excel_operator[col].apply(lambda s: " ".join(s.split()))

    df_excel_operator["EventTime"] = df_excel_operator["EventTime"].apply(lambda d: d.replace(".000000000","").replace("/","-"))
    df_excel_operator.to_csv(path_prcessed+output_flie_name, index=False)
    del df_excel_operator



print("> ------------ Operator Data Processing pre1 is complete. ")


# %%


