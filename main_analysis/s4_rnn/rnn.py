
# %%
from helpers.ploting import plotBargraph
import torch
from torch import nn
import numpy as np
import pickle
from sklearn import metrics
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt


#%%
def get_batches(arr1,arr2, batch_size, seq_length):
    '''Create a generator that returns batches of size
       batch_size x seq_length from arr.
       
       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''
    
    ## TODO: Get the number of batches we can make
    t = arr1.size//(batch_size * seq_length)
    
    n_batches =  t
    
    ## TODO: Keep only enough characters to make full batches
    arr1 =  arr1[:batch_size*(seq_length*n_batches)]
    arr2 =  arr2[:batch_size*(seq_length*n_batches)]

    ## TODO: Reshape into batch_size rows
    arr1 = arr1.reshape((batch_size,-1))
    arr2 = arr2.reshape((batch_size,-1))
    
    ## TODO: Iterate over the batches using a window of size seq_length
    for n in range(0, arr1.shape[1], seq_length):
        # The features
        x = arr1[:,n:n+seq_length]
        y = arr2[:,n:n+seq_length]
        yield x, y

#%%
dataset_dict = None

with open('seq_prof_suggestion.dataset', 'rb') as f: 
    dataset_dict = pickle.load(f)

vocab = dataset_dict["vocab"]
int2vocab = dataset_dict["int2vocab"]
vocab2int = dataset_dict["vocab2int"]
seq_length = dataset_dict["seq_length"]


train_inputs = dataset_dict["train_inputs"]
train_targets = dataset_dict["train_targets"]

valid_inputs = dataset_dict["valid_inputs"]
valid_targets = dataset_dict["valid_targets"]


# %%
batches = get_batches(np.array(train_inputs),np.array(train_targets),batch_size=2, seq_length=seq_length-1)
x, y = next(batches)

print(x.shape, y.shape)

# printing out the first 10 items in a sequence
print('x\n', x)
print('\ny\n', y)

# %% [markdown]
# ## RNN Network

# %%
# check if GPU is available
train_on_gpu = torch.cuda.is_available()
print(train_on_gpu)

if(train_on_gpu):
    print('Training on GPU!')
else: 
    print('No GPU available, training on CPU; consider making n_epochs very small.')


# %%
# class AlarmRNN1(nn.Module):
    
#     def __init__(self,tokens,int2vocab,vocab2int,embedding_dim,n_hidden=256, n_layers=2,
#                                drop_prob=0.5, lr=0.001):
#         super(AlarmRNN1,self).__init__()
#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.lr = lr
        
#         # creating character dictionaries
#         self.chars = tokens # vocab
#         # int2vocab = dict(enumerate(self.chars))
#         # vocab2int = {v:k for k,v in int2vocab.items()}

#         self.int2char = int2vocab #dict(enumerate(self.chars))
#         self.char2int = vocab2int  #{ch: ii for ii, ch in self.int2char.items()}
                
#         ## TODO: define the layers of the model
#         self.embedding = nn.Embedding(len(self.chars), embedding_dim)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,dropout=self.drop_prob, batch_first=True)
#         self.droput = nn.Dropout(p=self.drop_prob)
#         self.fc1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
#         self.relu1 = nn.ReLU()
#         self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
#         self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(in_features=self.n_hidden, out_features=len(self.chars)) 
#         self.softmax = nn.LogSoftmax(dim=1)  
    
#     def forward(self, x, hidden):
#         ''' Forward pass through the network. 
#             These inputs are x, and the hidden/cell state `hidden`. '''
                
#         ## TODO: Get the outputs and the new hidden state from the lstm
#         x = x.long()
#         embeds = self.embedding(x)
#         out, hidden = self.lstm(embeds,hidden)
#         out = self.droput(out)
#         # Contiguous variables: If you are stacking up multiple LSTM outputs, it may be necessary to use .contiguous() to reshape the output.
#         out = out.contiguous().view(-1,self.n_hidden)
#         out = self.fc1(out)
#         out = self.relu1(out)
#         out = self.fc2(out)
#         out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.softmax(out)
#         # return the final output and the hidden state
#         return out, hidden
    
    
#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
        
#         if (train_on_gpu):
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
#         return hidden

# class AlarmRNN2(nn.Module):
    
#     def __init__(self,tokens,int2vocab,vocab2int,embedding_dim,n_hidden=256, n_layers=2,
#                                drop_prob=0.5, lr=0.001):
#         super(AlarmRNN2,self).__init__()
#         self.drop_prob = drop_prob
#         self.n_layers = n_layers
#         self.n_hidden = n_hidden
#         self.lr = lr
        
#         # creating character dictionaries
#         self.chars = tokens # vocab
#         # int2vocab = dict(enumerate(self.chars))
#         # vocab2int = {v:k for k,v in int2vocab.items()}

#         self.int2char = int2vocab #dict(enumerate(self.chars))
#         self.char2int = vocab2int  #{ch: ii for ii, ch in self.int2char.items()}
                
#         ## TODO: define the layers of the model
#         self.embedding = nn.Embedding(len(self.chars), embedding_dim)
#         self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,dropout=self.drop_prob, batch_first=True)
#         self.droput = nn.Dropout(p=self.drop_prob)
#         # self.fc1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
#         # self.relu1 = nn.ReLU()
#         # self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
#         # self.relu2 = nn.ReLU()
#         self.fc3 = nn.Linear(in_features=self.n_hidden, out_features=len(self.chars)) 
#         self.softmax = nn.LogSoftmax(dim=1)  
    
#     def forward(self, x, hidden):
#         ''' Forward pass through the network. 
#             These inputs are x, and the hidden/cell state `hidden`. '''
                
#         ## TODO: Get the outputs and the new hidden state from the lstm
#         x = x.long()
#         embeds = self.embedding(x)
#         out, hidden = self.lstm(embeds,hidden)
#         out = self.droput(out)
#         # Contiguous variables: If you are stacking up multiple LSTM outputs, it may be necessary to use .contiguous() to reshape the output.
#         out = out.contiguous().view(-1,self.n_hidden)
#         # out = self.fc1(out)
#         # out = self.relu1(out)
#         # out = self.fc2(out)
#         # out = self.relu2(out)
#         out = self.fc3(out)
#         out = self.softmax(out)
#         # return the final output and the hidden state
#         return out, hidden
    
    
#     def init_hidden(self, batch_size):
#         ''' Initializes hidden state '''
#         # Create two new tensors with sizes n_layers x batch_size x n_hidden,
#         # initialized to zero, for hidden state and cell state of LSTM
#         weight = next(self.parameters()).data
        
#         if (train_on_gpu):
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
#                   weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
#         else:
#             hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
#                       weight.new(self.n_layers, batch_size, self.n_hidden).zero_())
        
#         return hidden

class AlarmRNN3(nn.Module):
    
    def __init__(self,tokens,int2vocab,vocab2int,embedding_dim,n_hidden=256, n_layers=2,
                               drop_prob=0.5, lr=0.001):
        super(AlarmRNN3,self).__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        
        # creating character dictionaries
        self.chars = tokens # vocab
        # int2vocab = dict(enumerate(self.chars))
        # vocab2int = {v:k for k,v in int2vocab.items()}

        self.int2char = int2vocab #dict(enumerate(self.chars))
        self.char2int = vocab2int  #{ch: ii for ii, ch in self.int2char.items()}
                
        ## TODO: define the layers of the model
        self.embedding = nn.Embedding(len(self.chars), embedding_dim)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=self.n_hidden, num_layers=self.n_layers,dropout=self.drop_prob, batch_first=True)
        self.droput = nn.Dropout(p=self.drop_prob)
        
        # self.fc1 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        # self.fc2 = nn.Linear(in_features=self.n_hidden, out_features=self.n_hidden)
        self.fc3 = nn.Linear(in_features=self.n_hidden, out_features=len(self.chars))
        
        # self.relu1 = nn.LeakyReLU()
        # self.relu2 = nn.LeakyReLU()
        # self.relu3 = nn.LeakyReLU()
        self.softmax = nn.LogSoftmax(dim=1)  
    
    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
                
        ## TODO: Get the outputs and the new hidden state from the lstm
        
        # print(x.shape)
        # print(hidden)
        x = x.long()
    

        embeds = self.embedding(x)
        # embeds = self.relu1(embeds)
        out, hidden = self.gru(embeds,hidden)
        # Contiguous variables: If you are stacking up multiple LSTM outputs, it may be necessary to use .contiguous() to reshape the output.
        out = out.contiguous().view(-1,self.n_hidden)
        
        # out = self.fc1(out)
        # out = self.droput(out)
        # out = self.relu2(out)
        # out = self.fc2(out)
        # out = self.relu3(out)
         
        out = self.fc3(out)
        out = self.softmax(out)
        # return the final output and the hidden state
        return out, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
         
        if (train_on_gpu):
            device = torch.device("cuda")
        else:
            device = torch.device("cpu") 
        

        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device)
        return hidden



# %%
import plotly.graph_objects as go

def plotConfusionMatrix(predictions, targets, labels):
    cm = metrics.confusion_matrix(targets, predictions, normalize="true", labels=labels)
    
    cm = cm * 100 # for percentage
    
    data = [[None for i in range(cm.shape[0])] for j in range(cm.shape[0])]

    if len(data) == 0:
        print(" --------------> Heatmap:no data exist in heatmap")

    # print(">> Dimension", len(data[0]), len(data))

    more_than = []
    all_alarms = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[0]):
            if cm[i,j] > 0.01:
                data[i][j] = cm[i,j]

            if cm[i,j] >= 65 and i ==j:
                more_than.append((i,j, cm[i,j]))
            
            if i ==j:
                all_alarms.append((i,j, cm[i,j]))
    
    
    print(f">> Alarms Accuracy: {[a[2] for a in sorted(all_alarms, key=lambda arg: arg[2])] }")
    
    snames_sub = [f"S{i}" for i,j,val in all_alarms]
    trace = go.Bar(x=snames_sub,y=[val for _,_, val in all_alarms])
    
    fig = go.Figure(data=trace) 
    # # updating the figure properties
    # fig.update_xaxes(title_text="Source Name")
    # fig.update_yaxes(title_text="Accuracy (%)", range=[0,100])
    fig.update_layout(
    # xaxis_tickfont_size=14,
    yaxis=dict(
        title='Accuracy (%)',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Source Name",
        titlefont_size=16,
        tickfont_size=14,
    ),
    legend=dict(
        x=0,
        y=1.0,
        bgcolor='rgba(255, 255, 255, 0)',
        bordercolor='rgba(255, 255, 255, 0)'
    ),
    # barmode='group',
    height=600, 
    width=1200,
    template='seaborn', # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
    fig.show()

    print(f">> Total Source= {len(labels)}, Sources whose accuracies are more than 65% = {len(more_than)}")


    # HeatMap

    # temp_dat = [[1.0, 0.0, 0.0  ],[0.,   0.62, 0.38],[0.,   0.,   1.  ]]

    fig = go.Figure(data=go.Heatmap(
        z= data,
        colorscale='Greys',
        x = [f"S{i}" for i in range (len(labels))],
        y = [f"S{i}" for i in range (len(labels))],
        hoverongaps=False, 
        hovertemplate=None
    ))
    fig.update_layout(
        width=1200,
        height=1200,
        # xaxis_nticks=cm.shape[0],
        # yaxis_nticks=cm.shape[0],

        yaxis=dict(
        title='Predicted Label',
        titlefont_size=16,
        tickfont_size=14,
        ),
        xaxis=dict(
        title = "True Label",
        titlefont_size=16,
        tickfont_size=14,
        )
        
    )
    fig.show()


def validationReport(net, predictions, targets, ignore):

    predictions = [net.int2char[id] for id in predictions.numpy()]
    targets = [net.int2char[id] for id in targets.numpy()]

    labels = [v for k,v in net.int2char.items() if k != ignore]


    plotConfusionMatrix(predictions, targets,labels)
    report = metrics.classification_report(targets,predictions,labels=labels)
    # print(report)

def train(m_name,net,data,seq_length,batch_size,epochs=10, lr=0.001,wieght_decay=0,clip=5, print_every=10, report_every=10):
    ''' Training a network 
    
        Arguments
        ---------
        
        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss
    
    '''
    print(f"Batch Size ={batch_size}, seq_length={seq_length}")

    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wieght_decay)
    
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss(ignore_index=net.char2int["NoName"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min',patience=6,verbose=True, factor=0.5)
    
    if(train_on_gpu):
        net.cuda()
    
    loss = 0

    all_epochs_train_lossess = []
    all_epochs_valid_lossess = [] 

    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        train_losses = []
        for x, y in get_batches(data["train_inputs"], data["train_targets"], batch_size, seq_length):

            train_inputs, train_targets = torch.from_numpy(x), torch.from_numpy(y)
            
            if(train_on_gpu):
                train_inputs, train_targets = train_inputs.cuda(), train_targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            #h = tuple([each.data for each in h]) # for lstm
            h = h.data # for GRU
            # zero accumulated gradients
            net.zero_grad()
            
            # get the output from the model
            train_output, h = net(train_inputs, h)
        
            loss = criterion(train_output, train_targets.view(batch_size*seq_length).long())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()
            train_losses.append(loss.item())
            
        # loss stats with validation 
        if (e+1) % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            valid_predictions = None  #torch.empty()
            valid_targets =  None #torch.empty()
            for x, y in get_batches(data["valid_inputs"],data["valid_targets"],batch_size, seq_length):
                
                x, y = torch.from_numpy(x), torch.from_numpy(y)
                val_inputs, val_targets = x, y

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                # val_h = tuple([each.data for each in val_h]) # for lstm
                val_h = val_h.data
                    
                if(train_on_gpu):
                    val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()

                val_output, val_h = net(val_inputs, val_h)

                _, predictions = torch.max(val_output,dim=1) # probs are the indexes
                predictions = predictions.to("cpu")
                targets = val_targets.view(batch_size*seq_length).long().to('cpu')
                
                if valid_predictions is not None:
                    valid_predictions = torch.cat((valid_predictions,predictions))
                    valid_targets = torch.cat((valid_targets,targets))
                else:
                    valid_predictions = predictions
                    valid_targets = targets


                val_loss = criterion(val_output, val_targets.view(batch_size*seq_length).long())
                val_losses.append(val_loss.item())
            
            print(f"> Epoch: {e+1}/{epochs}...Train Loss: {np.mean(train_losses):.4f}... Val Loss: {np.mean(val_losses):.4f}")

            all_epochs_train_lossess.append(np.mean(train_losses))
            all_epochs_valid_lossess.append(np.mean(val_losses))
            
            # if np.mean(val_losses) < 1.2 and np.mean(val_losses) > 0.75:
            #     torch.save(net,f"model_{m_name}_val_loss_{np.mean(val_losses):.2f}.pth")
            if np.mean(val_losses) < 1.2:
                torch.save(net,f"model_{m_name}_val_loss{np.mean(val_losses):.2f}.pth")     

            
            if (e+1)% report_every==0:
                validationReport(net=net, predictions=valid_predictions, targets=valid_targets,ignore=net.char2int["NoName"] )
           
            scheduler.step(np.mean(val_losses)) # to reduce lr
            net.train() # reset to train mode after iterationg through validation data


    return all_epochs_train_lossess, all_epochs_valid_lossess

#%%
## TODO: set your model hyperparameters

torch.cuda.empty_cache()

n_hidden=1024
n_layers=5
n_epochs = 200 # start small if you are just testing initial behavior
batch_size = 64
embedding_dim = 512
drop_prob = 0.1
#    batch_size x seq_length from arr.

data = {"train_inputs":np.array(train_inputs), "train_targets": np.array(train_targets), "valid_inputs":np.array(valid_inputs), "valid_targets":np.array(valid_targets)}

print("Here")
net3 = AlarmRNN3(tokens=vocab,int2vocab=int2vocab,vocab2int=vocab2int,embedding_dim=embedding_dim,n_hidden=n_hidden,n_layers=n_layers,drop_prob=drop_prob)
t2_loss, v2_loss=train("GRU_3_prof",net3,data, seq_length=seq_length-1, batch_size=batch_size, epochs=n_epochs,lr=0.000001, wieght_decay = 0.0000001, print_every=1,report_every=10)


#%%
_, _ =train("GRU_3_prof",net3,data, seq_length=seq_length-1, batch_size=batch_size, epochs=200,lr=0.00005, wieght_decay = 0.000003, print_every=1,report_every=10)


#%%

losses_dict = {"train-loss": t2_loss, "valid-loss":v2_loss}
with open('gru.losses', 'wb') as f:
  pickle.dump(losses_dict , f)

print(">> lossses saved")

# %%

# n_layers =3
# net2 = AlarmRNN1(tokens=vocab,int2vocab=int2vocab,vocab2int=vocab2int,embedding_dim=embedding_dim,n_hidden=n_hidden,n_layers=n_layers,drop_prob=drop_prob)
# t2_loss, v2_loss=train("lstm_3",net2,data, seq_length=seq_length-1, batch_size=batch_size, epochs=n_epochs,lr=0.001, wieght_decay = 0.0002, print_every=1,report_every=10)

# t2_loss, v2_loss
#%%
# torch.save(net3,"model_GRU_3_val_loss_1_Accuracy_better.pth")

# n_layers =3

# model = torch.load("model_GRU_3_val_loss_0.86.pth")
# batch_size=64
# t3_loss, v3_loss=train("GRU_3",model,data, seq_length=seq_length-1, batch_size=batch_size, epochs=n_epochs,lr=0.0001, wieght_decay = 0.0003, print_every=1,report_every=10)
# t3_loss, v3_loss

 # %%

# batch_size = 64
# n_epochs = 400
# model = torch.load("rnn2_fine_tuned_0.80_loss.pth")
# #  model

# train(model,data, seq_length=seq_length-1, batch_size=batch_size, epochs=n_epochs,lr=0.001/4, wieght_decay = 0.0007, print_every=1,report_every=40)
# torch.save(model,"rnn2_fine_tuned_0.80_loss_2.pth")


# %%



import pickle
import plotly.graph_objects as go

dict_loss = None
with open('gru.losses', 'rb') as f: 
    dict_loss = pickle.load(f)

fig = go.Figure()
x= list(range(len(dict_loss["train-loss"])))
fig.add_trace(go.Scatter(x=x[:200], y=dict_loss["valid-loss"][:200], name="linear",line_shape='linear'))
# fig.add_trace(go.Scatter(x=x, y= dict_loss["train-loss"], name="linear",line_shape='linear'))

fig.update_layout(
    # xaxis_tickfont_size=14,
    yaxis=dict(
        title='Validation Loss',
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis=dict(
        title = "Number of Epochs",
        titlefont_size=16,
        tickfont_size=14,
    ),
    xaxis_nticks=20,
    # barmode='group',
    height=400, 
    width=800,
    template='seaborn' # ggplot2
    # bargap=0. # gap between bars of adjacent location coordinates.
    # bargroupgap=0.1 # gap between bars of the same location coordinate.
    )
fig.show()

# list(range(len(dict_loss["train-loss"])))

# %% 

"""
    Evaluate the model
"""

model = torch.load("model_GRU_3_prof_val_loss1.04.pth")

model.eval()

# Get validation loss
val_h = model.init_hidden(batch_size)
val_losses = []
model.eval()
valid_predictions = None  #torch.empty()
valid_targets =  None #torch.empty()
for x, y in get_batches(data["valid_inputs"],data["valid_targets"],batch_size, seq_length):
    
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    val_inputs, val_targets = x, y

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    # val_h = tuple([each.data for each in val_h]) # for lstm
    val_h = val_h.data
        
    if(train_on_gpu):
        val_inputs, val_targets = val_inputs.cuda(), val_targets.cuda()

    val_output, val_h = model(val_inputs, val_h)

    _, predictions = torch.max(val_output,dim=1) # probs are the indexes
    predictions = predictions.to("cpu")
    targets = val_targets.view(batch_size*seq_length).long().to('cpu')
    if valid_predictions is not None:
        valid_predictions = torch.cat((valid_predictions,predictions))
        valid_targets = torch.cat((valid_targets,targets))
    else:
        valid_predictions = predictions
        valid_targets = targets


    # val_loss = criterion(val_output, val_targets.view(batch_size*seq_length).long())
    # val_losses.append(val_loss.item())

    # print(f"> Val Loss: {np.mean(val_losses):.4f}")
validationReport(net=model, predictions=valid_predictions, targets=valid_targets,ignore=model.char2int["NoName"] )


# %%
