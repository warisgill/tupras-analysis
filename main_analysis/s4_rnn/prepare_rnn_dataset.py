#%%
from datetime import timedelta
from helpers import alarms
import time
import numpy as np
from functools import partial
from sklearn.model_selection import train_test_split
import itertools
import pickle

# %%

"""  Lodading the Data and Preprocessing """
PATH = "/home/waris/Github/tupras-analysis/data/"
alarm_file_path = PATH + "processed/alarms/final/final-all-months-alarms.csv"
# op_action_file_path = PATH + "processed/operator-actions/final/final-all-month-actions.csv"

start = time.time()

df_main_alarms =alarms.loadAlarmsData(file_path=alarm_file_path)
# df_main_actions = loadOperatorData(file_path=op_action_file_path)

""" Common Sources in all months. Try it but can be skipped. """
df_main_alarms = alarms.getDFWithCommonSourcesInAllMonths(df_main_alarms)

""" Chaning name 2 alias for alarm data but skipping it """
# source2Alias, alias2source = convertSourceNamesToAlias(df_main_alarms)

print("Total Time to load the data ", time.time()-start)
# df_main_alarms

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
ignore_comm_alarms = True
momentary_alarms_f = 0  # seconds
staling_alarm_f = (60*60) * 12 # hours
min_alarms_per_source_f = 200 # any source which is not triggered atleast 20 times in whole dataset will be removed
months_f = df_main_alarms["Year-Month"].unique()
snames_f = [] # ONLY USE IF NOT IGNORING COMM ALRMS

df_alarms_new = alarms.filterAlarmData(df_main_alarms, months=months_f, sources_filter=snames_f,
                                     monmentarly_filter=momentary_alarms_f, staling_filter=staling_alarm_f, ingore_communication_alarms=ignore_comm_alarms, min_alarms_per_source=min_alarms_per_source_f)


df_alarms_new

#%%

df_rnn = alarms.removeChatteringAlarms(df_alarms_new)

df_rnn

#%%
source2count = dict(df_rnn["SourceName"].value_counts())
print(len(source2count))
print(source2count)

# %%
def getSequenceByMonth(df,month,seq_duration_gap,filter_short_seq):
    print(f">> Seq By Month:{month}, duration to next seq: {seq_duration_gap}, ignore seq len: {filter_short_seq}")

    list_of_sequences = []
    # max_len = 0
    # min_len = 9999   
    df_month = df[df["Year-Month"]==month]
    assert 1 == len(df_month["Year-Month"].unique())
    
    alarms= df_month.to_dict(orient="records")
    alarms = [alarm for alarm in sorted(alarms, key=lambda arg: arg["StartTime"], reverse=False)] # sorting
    # print(alarms[:2])
    i =0
    j= 0
    while i <len(alarms):
        prev_start = alarms[i]["StartTime"]
        seq = []
        seq.append(alarms[i])
        j = i+1
        while j < len(alarms):    
            next_start = alarms[j]["StartTime"]
            delta = timedelta.total_seconds(next_start - prev_start)
            # print(delta)
            assert delta >= 0
            if delta >= seq_duration_gap:
                break

            seq.append(alarms[j])
            j += 1
        i = j
        # if len(seq)>max_len:
        #     max_len = len(seq)
        # if len(seq)<min_len:
        #     min_len = len(seq)
        

        if len(seq)>=filter_short_seq:
            seq = [alarm for alarm in sorted(seq, key=lambda arg: arg["StartTime"], reverse=False)]
            seq = [alarm["SourceName"] for alarm in seq]
            list_of_sequences.append(seq)
    
    # list_of_sequences = [l for l in list_of_sequences if (len(l)>filter_short_alarms)]
    # print(f">> Min Seq Len: {min_len}, max seq len: {max_len}")
    return list_of_sequences


#Doing Padding
def padding(seq_length,arr):
    
    if len(arr) < seq_length:
        return ["NoName" for i in range(seq_length-len(arr))] + arr
    # elif len(arr) > seq_length: # if bigger than seq_length use earliers
    #     return arr[0:seq_length]

    return arr

def getAllMonths2Seqs(df,seq_time_gap,seq_ignore_len):
    # print(f">> All Months: Seq_len = {seq_length}")
    month2alarms = {}

    all_seqs = [] 

    for month in df["Year-Month"].unique():
        li_of_seqs = getSequenceByMonth(df,month,seq_time_gap,seq_ignore_len)
        all_seqs = all_seqs + li_of_seqs
        # seq_length = max([len(seq) for seq in li_of_seqs]) 
        # f = partial(padding,seq_length)
        month2alarms[month] =  all_seqs #[f for l in li_of_seqs]
    
    max_seq_len = max([len(seq) for seq in all_seqs])
    f = partial(padding,max_seq_len)

    month2alarms = {k:[f(l) for l in v] for k,v in month2alarms.items()}

    return month2alarms, max_seq_len


def getTrainAndValidationData(df,seq_time_gap,seq_ignore_len,test_size,shuffle=False):
    print(f">> Spliting Data Size (valid%): {test_size}")
    train_data = []
    valid_data = []
    months2seq, max_seq_len = getAllMonths2Seqs(df,seq_time_gap=seq_time_gap,seq_ignore_len=seq_ignore_len)

    for month in df["Year-Month"].unique():
      train,valid = train_test_split(months2seq[month],test_size=test_size,shuffle=shuffle)
      train_data += train
      valid_data += valid

    return train_data, valid_data, max_seq_len  

# def encodeData(mydict,l):
#     return [mydict[e] for e in l]

# def getInputsAndTargets(encoded_alarms):
#     input_seqs_train = []
#     target_seqs_train = []

#     for i in range(len(encoded_alarms)):
#         # remove the last char from input seq
#         input_seqs_train.append(encoded_alarms[i][:-1])

#         # remove the first char from input seq
#         target_seqs_train.append(encoded_alarms[i][1:])
#         # print(f"orignal={encoded_alarms[i]} \n Input Seq={input_seqs_train[i]}, \n Target Seq = {target_seqs_train[i]}")

#     inputs = []
#     targets = []
#     for l in input_seqs_train:
#         inputs = inputs+l

#     for l in target_seqs_train:
#         targets = targets + l


#     # for row in range(len(input_seqs_train)): 
#     row = 0

#     # print(inputs[row*(seq_length-1):row*(seq_length-1)+ seq_length-1],targets[row*(seq_length-1):row*(seq_length-1)+ seq_length-1])
#     print(input_seqs_train[row], target_seqs_train[row])
#     return inputs, targets, input_seqs_train, target_seqs_train



# %% [markdown]
# ## Mention the Sequence Length => how many alarms you want to put in a sequence
# Any change need to rerun from here

# %%

ingore_short_seq_len = 3 
# long_len = seq_length 
duration_from_1_seq_to_next = 60*15 # duration in seconds
test_size = 0.1
shuffle = False

train_data, valid_data, max_seq_len = getTrainAndValidationData(df_rnn,seq_time_gap=duration_from_1_seq_to_next,seq_ignore_len=ingore_short_seq_len,test_size=test_size,shuffle=shuffle)
seq_length = max_seq_len

print(f">> Train Size: {len(train_data)}, Validation Size: {len(valid_data)}")  
print(valid_data[1])


#%%

store_dict = {"seq-duration":duration_from_1_seq_to_next, "train": train_data, "valid":valid_data, "sequence-length":seq_length}

with open('raw-dataset-15-mins.dataset', 'wb') as f:
  pickle.dump(store_dict , f)

print(">> Dataset is stored")

# %% [markdown]
# # ## Preparing Vocab and conversion from vocab2int and into2vocab

# # %%

vocab = set(list(itertools.chain.from_iterable(train_data+valid_data)))
int2vocab = dict(enumerate(vocab))
vocab2int = {v:k for k,v in int2vocab.items()}
assert len(vocab2int) == len(vocab)

f = partial(encodeData,vocab2int)
encoded_train_alarms = [f(l) for l in train_data]
encoded_valid_alarms = [f(l) for l in valid_data]


train_features = np.array(encoded_train_alarms)
train_features.shape

# %% [markdown]
# ## Preparing Data To Use mini batches

# %%

row = 0 # this row should always be zero because in function it is zero

print(">> ============ Training Set ==============")
train_inputs, train_targets, _, _ = getInputsAndTargets(encoded_train_alarms)
print(train_inputs[row*(seq_length-1):row*(seq_length-1)+ seq_length-1],train_targets[row*(seq_length-1):row*(seq_length-1)+ seq_length-1])

print(">> =========== Validation Set ==============")
valid_inputs, valid_targets, _, _ = getInputsAndTargets(encoded_valid_alarms)    
print(valid_inputs[row*(seq_length-1):row*(seq_length-1)+ seq_length-1],valid_targets[row*(seq_length-1):row*(seq_length-1)+ seq_length-1])

print(">> Ignore Seq", vocab2int["NoName"])

# %% [markdown]
"""
    Use pickling to store dataset.
"""

import pickle

store_dataset = {
"vocab": vocab,
"int2vocab":int2vocab,
"vocab2int":vocab2int,
"seq_length": seq_length,
"train_inputs":train_inputs, 
"train_targets":train_targets, 
"valid_inputs": valid_inputs, 
"valid_targets": valid_targets
}

with open('seq.dataset', 'wb') as f:
  pickle.dump(store_dataset , f)

print(">> Dataset is stored")