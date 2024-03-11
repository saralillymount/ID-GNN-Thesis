# %% [markdown]
# # hash-GNN

# %% [markdown]
# ## Setup

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import networkx as nx
import dgl
from pathlib import Path
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GINConv, GraphConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
import hashlib
import argparse
from datetime import datetime

# %% [markdown]
# # Class Defs

# %% [markdown]
# ## Loader

# %%
class loader:    
    def __init__(self):
        self.graphid=0
        self.getGraphid={}
        self.getNodesOfGraphId={}
        self.countID=0
        self.G={}
        self.co={}
        self.revco={}
    
    def nodeID(self,x):
        if x not in self.co:
            self.co[x]=self.countID
            self.countID=self.countID+1
            self.revco[self.co[x]]=x
        return self.co[x]
    
    def read(self,file): # file is csv edgelist
        x=pd.read_csv(file,sep=',').values
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,0])
            j=self.nodeID(x[a,1])
            self.addEdge((i,j))
        self.fixG()
        
    def readWithFeatures(self,edgeFile,featFile):
        x=pd.read_csv(edgeFile,sep=',',header=None).values
        print('Reading file:', edgeFile)
        print('Reading file:', edgeFile, file=logfile)       
        print('Reading file:', featFile)
        print('Reading file:', featFile, file=logfile)
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,0])
            j=self.nodeID(x[a,1])
            self.addEdge((i,j))
        self.fixG()
        features=pd.read_csv(featFile,sep=',',header=None).values
        ids={features[x,0]:x for x in range(features.shape[0])}
        ids=[ids[self.revco[x]] for x in range(len(self.G))]
        self.features=features[ids,1:]

    def readWithFeaturesHash(self,edgeFile,featFile,hashFile):
        x=pd.read_csv(edgeFile,sep=',',header=None).values       
        for a in range(x.shape[0]):
            i=self.nodeID(x[a,0])
            j=self.nodeID(x[a,1])
            self.addEdge((i,j))
        self.fixG()
        features=pd.read_csv(featFile,sep=',',header=None).values
        ids={features[x,0]:x for x in range(features.shape[0])}
        ids=[ids[self.revco[x]] for x in range(len(self.G))]
        self.features=features[ids,1:]

        hashFeatures=pd.read_csv(hashFile,sep=',',header=None)
        hashFeatures = hashFeatures.values
        hids={hashFeatures[x,0]:x for x in range(hashFeatures.shape[0])}
        hids=[hids[self.revco[x]] for x in range(len(self.G))]
        self.hashFeatures=hashFeatures[hids,1:]
    
    def getNodes(self,gid):
        return self.getNodesOfGraphId[gid]
        
    def storeEmb(self,file,data):
        file1 = open(file, 'w') 
        for a in range(data.shape[0]):
            s=''+str(int(self.revco[a]))
            for b in range(data.shape[1]):
                s+=','+str(data[a,b])
            file1.write(s+"\n")
        file1.close()            
    
    def fixG(self):
        for g in range(len(self.G)):
            self.G[g]=np.array([x for x in self.G[g]])

    def addEdge(self,s):
        (l1,l2)=s
        if l1 not in self.G:
            self.G[l1]=set()
        if l2 not in self.G:
            self.G[l2]=set()
        self.G[l1].add(l2)
        self.G[l2].add(l1) #undirected?
    
    def getDegreeMtx(self):
        return np.array([[self.G[x].shape[0] for y in range(2)]for x in self.G])


# %%
def sampling(g,depth,k=1,p=0.1): ##used to get le and le1
    li=[]
    li1=[]
    for p in range(k): 
        for x in range(len(g)):
            #print(x)
            initial=x
            count=0
            while count<depth:
                if np.random.rand()<p:
                    break
                initial=np.random.choice(g[initial])
                count+=1
            li.append(initial)
            li1.append(x)
    return np.array(li),np.array(li1) ##nodes' neighbors, nodes

# %%
def createHash(features_df):
    ## Calculate hashFeatures    
    featureString = features_df.astype(str).apply(''.join, axis=1)
    # print(featureString)
    if (hashType == 'SHA256'):
        hashString = featureString.apply(
            lambda x: 
                "{0:0256b}".format(int(hashlib.sha256(x.encode()).hexdigest(), 16))
        )
    # elif (hashType == 'MD5'):
    #     hashString = featureString.apply(
    #         lambda x: 
    #             "{0:0128b}".format(int(hashlib.md5(x.encode()).hexdigest(), 16))
    #     )
    elif (hashType == 'SHA3'):
        hashString = featureString.apply(
            lambda x: 
                "{0:0256b}".format(int(hashlib.sha3_256(x.encode()).hexdigest(), 16))
        )   
    else:
        print('problem encountered')

    # ############## Hash again ###############################
    # if (hashType == 'SHA256'):
    #     hashString = hashString.apply(
    #         lambda x: 
    #             "{0:0256b}".format(int(hashlib.sha256(x.encode()).hexdigest(), 16))
    #     )
    # elif (hashType == 'MD5'):
    #     hashString = hashString.apply(
    #         lambda x: 
    #             "{0:0128b}".format(int(hashlib.md5(x.encode()).hexdigest(), 16))
    #     )
    # elif (hashType == 'SHA3'):
    #     hashString = hashString.apply(
    #         lambda x: 
    #             "{0:0256b}".format(int(hashlib.sha3_256(x.encode()).hexdigest(), 16))
    #     )   
    # else:
    #     print('problem encountered')
    # print(hashString)

    hashFeatures = hashString.apply(lambda x: pd.Series(list(x)))
    hashFeatures = hashFeatures.astype(int).values
    # print(hashFeatures)

    maxh=np.max(hashFeatures)
    hashFeatures = hashFeatures/maxh
    hashFeatures = torch.FloatTensor(hashFeatures)
    if cudaIsTrue:
        hashFeatures = hashFeatures.cuda(device)
    return hashFeatures

# %% [markdown]
# ## Modules

# %%
class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and sigmoid."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)
        self.activ= 'sigmoid'

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        if (self.activ == 'sigmoid'):
            h = torch.sigmoid(h)
        elif (self.activ == 'relu'):
            h = torch.relu(h)
        return h

class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, activation):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.hidd=output_dim
        self.activ = activation

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                if (self.activ == 'relu'):
                    h = torch.relu(self.batch_norms[i](self.linears[i](h)))
                elif (self.activ == 'sigmoid'):
                    h = torch.sigmoid(self.batch_norms[i](self.linears[i](h)))
                else:
                    h = self.batch_norms[i](self.linears[i](h))
            return self.linears[-1](h)

class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 output_dim, final_dropout, learn_eps, graph_pooling_type,
                 neighbor_pooling_type, activation):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)

        """
        super(GIN, self).__init__()
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        self.activ = activation

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim, self.activ) ## use same activation as GIN
            else: 
                mlp = MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim, self.activ) ## use same activation as as GIN

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()
        self.linearcomp=nn.Linear( (num_layers-1)*(hidden_dim), output_dim)
        
        # print(num_layers*hidden_dim, output_dim)
        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        #hidden_rep = [h]
        ll=[]
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            if (self.activ=='relu'):
                h = torch.relu(h)
            elif(self.activ=='sigmoid'):
                h = torch.sigmoid(h)
            ll.append(h)
            #hidden_rep.append(h)

        #score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        #for i, h in enumerate(hidden_rep):
           # pooled_h = self.pool(g, h)
           # score_over_layer += self.drop(self.linears_prediction[i](pooled_h))

        v= torch.cat(ll, dim=-1)
        #print(v.shape)
        return self.linearcomp(v)

class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        self.n_layers = n_layers
        self.activ = activation
        if (self.activ == 'relu'):
            self.layerActiv = torch.relu
        elif (self.activ == 'sigmoid'):
            self.layerActiv = torch.sigmoid
        else:
            self.layerActiv = None
        # input layer
        self.layers.append(GraphConv(in_feats, n_hidden, activation=self.layerActiv))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=self.layerActiv))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)
        self.linearcomp=nn.Linear( (n_layers)*(n_hidden), n_classes)

    # def forward(self, g, features): ## previous
    #     h = features
    #     ll=[]
    #     for i in range(self.n_layers):
    #         if i != 0:
    #             h = self.dropout(h)
    #         h = self.layers[i](g, h)
    #         # ll.append(h) ##
    #     v = h #torch.cat(ll, dim=-1)
    #     #return h
    #     return v #self.linearcomp(v)

    def forward(self, g, features): #from GIN-GNN
        h = features
        ll=[]
        for i in range(self.n_layers):
            if i != 0:
                h = self.dropout(h)
            h = self.layers[i](g, h)
            if (self.activ=='relu'):
                h = torch.relu(h)
            elif(self.activ=='sigmoid'):
                h = torch.sigmoid(h)
            ll.append(h) ##
        v = torch.cat(ll, dim=-1)
        #return h
        return self.linearcomp(v)

class Training(nn.Module):
    """GIN or GCN model"""
    def __init__(self,nlayers,featuresdim,hashdim,hiddendim,activation2,activationMLP,activationTrain):
        super(Training, self).__init__()
        # self.gin=GIN(num_layers=nlayers, num_mlp_layers=2, input_dim=degreedim, hidden_dim=hiddendim,
        #          output_dim=hiddendim,final_dropout=0.3,learn_eps=False,graph_pooling_type='sum',
        #          neighbor_pooling_type='sum')
        if(NNType == 'GCN'):
            self.gnn=GCN(in_feats=featuresdim+hashdim, n_hidden=hiddendim, n_classes=hiddendim, n_layers=nlayers, activation=activation2 ,dropout=0.3)
        elif(NNType == 'GIN'):   
            self.gnn=GIN(num_layers=nlayers, num_mlp_layers=2, input_dim=featuresdim+hashdim, hidden_dim=hiddendim,  #input_dim=featuresdim
                 output_dim=hiddendim,final_dropout=0.3,learn_eps=False,graph_pooling_type='sum',
                 neighbor_pooling_type='sum',activation=activation2)
        self.mlp= MLP(num_layers=2,input_dim=(hiddendim+hashdim), hidden_dim=(hiddendim+hashdim), output_dim=featuresdim,activation=activationMLP) 
        self.activ = activationTrain

    def forward(self, adjacency_matrix, features, hashFeatures, ids,ids1):  #logits = model(adj,noisyFeatures,hashFeatures,le,le1)
        # list of hidden representation at each layer (including input)
        #hidden_rep = [features]
        g0=torch.cat((hashFeatures, features), 1) # hash + X
        if(NNType == 'GCN'):
            if (cudaIsTrue):
                dgraph = dgl.from_networkx(nx.Graph(graph)).to(device)
            else:
                dgraph = dgl.from_networkx(nx.Graph(graph))

            g1=self.gnn(dgraph,g0)
        elif(NNType == 'GIN'):
            g1=self.gnn(adjacency_matrix,g0)
        g2=torch.cat((g1[ids1], hashFeatures[ids]), 1)
        g3=self.mlp(g2)
        if (self.activ == 'relu'):
            features = torch.relu(g3)
        elif(self.activ == 'sigmoid'):
            features = torch.sigmoid(g3)
        else:
            features = g3
        return features
    

# %% [markdown]
# # Loading Data

# %%
def load(data): # Load Train Data
    loadert=loader()
    edgefile = 'datasets/'+ data+'/data/'+data+'_edgelist.csv'
    featfile = 'datasets/'+ data+'/data/'+data+'_features.csv'
    loadert.readWithFeatures(edgefile, featfile)
    degree = loadert.getDegreeMtx()
    maxd = np.max(degree)
    degree = degree/maxd
    features = loadert.features
    maxf=np.max(features)
    features=features/maxf
    loaderlist=[]
    for x in range(len(loadert.G)):
        for y in loadert.G[x]:
            loaderlist.append((x,y))
    G = nx.Graph()
    G.add_edges_from(loaderlist)
    adj = nx.adjacency_matrix(G).tocsr()
    graph=loadert.G 
    features=torch.FloatTensor(features)
    degree = torch.FloatTensor(degree)
    adj = dgl.from_networkx(G)
    
    print('Reading Files: ' + edgefile + ', ' + featfile)# + ', ' + hashfile)###
    print('Reading Files: ' + edgefile + ', ' + featfile, file= logfile)###
    print(notes)
    print(notes, file=logfile)
    return loadert, features, adj, G, graph, degree

# %% [markdown]
# # Training

# %%
def do_training(data, features, degree, adj, G, graph, activation2, activationMLP, activationTrain, noiseNum, mu,sigma):    
    lr = 0.001
    patience = 300
    l2_coef = 0.0
    sparse=True
    nlayers=GNNLayers
    featuresdim=features.shape[1]
    hashdim = 256
    hiddendim=hiddendimension #XXX 2000 until 9-4-23
    print('learning rate: ', lr, ', patience: ', patience, ', l2_coef: ', l2_coef, ', nlayers: ', nlayers, ', hiddenDim: ', hiddendim, )
    print('learning rate: ', lr, ', patience: ', patience, ', l2_coef: ', l2_coef, ', nlayers: ', nlayers, ', hiddenDim: ', hiddendim, file=logfile)
    model = Training(nlayers,featuresdim,hashdim,hiddendim,activation2,activationMLP,activationTrain)

    model=model.cuda(device)
    print('Using CUDA')
    model.cuda(device)
    features = features.cuda(device)
    # hashFeatures = hashFeatures.cuda(device)###
    degree=degree.cuda(device)
    adj = adj.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
    lossf=nn.MSELoss()
    # lossf=nn.BCELoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0
    if 'load' in trainType:
        return model
    print('hash used: ', hashType)
    print('hash used: ', hashType, file= logfile)
    print('Mu/sigma/noiseNum: : ' + str(mu) + ', ' + str(sigma)+ ', ' +str(noiseNum))
    print('Mu/sigma/noiseNum: : ' + str(mu) + ', ' + str(sigma)+ ', ' +str(noiseNum), file=logfile)
    print('Loss used: ', lossf)
    print('Loss used: ', lossf, file= logfile)
    print('Activations: GNN-' ,model.gnn.activ, ', MLP-' ,model.mlp.activ, ', Train-',model.activ)
    print('Activations: GNN-' ,model.gnn.activ,', MLP-' ,model.mlp.activ, ', Train-',model.activ, file=logfile)

    model_name = 'datasets/'+ data+'/models/best_'+data+'_hash-'+NNType+'_'+hashType+'.pkl'
    ## Make noise
    noiseList = []
    noiseList.append(np.zeros([features.shape[0],features.shape[1]]))
    if (noiseNum >0):
        for i in range(noiseNum-1): 
            noise = np.random.normal(mu, sigma, [features.shape[0],features.shape[1]])
            noiseList.append(noise)

    for epoch in range(nb_epochs):
        #############################################################
        features_df = pd.DataFrame(features.cpu())
        noisyFeatures_df = features_df
        if (noiseNum >0):
            # Add Noise        
            noisyFeatures_df = features_df + noiseList[epoch % noiseNum]
            noisyFeatures = noisyFeatures_df.values
            noisyFeatures = torch.FloatTensor(noisyFeatures)
        else:
            noisyFeatures = features
        if (cudaIsTrue):
            noisyFeatures = noisyFeatures.cuda(device)
        
        ## Calculate hashfeatures  
        hashFeatures = createHash(noisyFeatures_df)
        hashFeatures = hashFeatures.cuda(device)
        ############################################################
        
        le,le1=sampling(graph,4,2,0.1)
        model.train()
        optimiser.zero_grad()
        logits = model(adj,noisyFeatures,hashFeatures,le,le1) #adjacency_matrix, features, hashFeatures, ids,ids1
        loss= lossf(logits, noisyFeatures[le])
        # print('Epoch:', epoch,' Loss:', loss)
        print('Epoch:', epoch,' Loss:', loss, file=logfile)
        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), model_name)
        else:
            cnt_wait += 1
        if cnt_wait == patience:
            print('Early stopping!')
            print('Early stopping!', file = logfile)
            print('Model state saved as: ' + model_name)
            print('Model state saved as: ' + model_name, file=logfile)
            break
        loss.backward()
        optimiser.step()
    print('Best_t = ' , best_t)
    print('Best_t = ' , best_t, file = logfile)
    return model



# %% [markdown]
# ## Store Emedding

# %%
def store(data, model, loadert, degree, adj, features, graph, activation2, activationMLP, activationTrain):
    model.load_state_dict(torch.load('datasets/'+ data+'/models/best_'+data+'_hash-'+NNType+'_'+hashType+'.pkl'))
    hashedFeatures = createHash(pd.DataFrame(features.cpu()))
    embedder = 'hash'
    embed_data(embedder, loadert, degree, adj, features, hashedFeatures, graph)
    embedder = 'GNN'
    embed_data(embedder, loadert, degree, adj, features, hashedFeatures, graph)
    embedder = 'hash+GNN'
    embed_data(embedder, loadert, degree, adj, features, hashedFeatures, graph)
# %% [markdown]
# ## Embed Function

# %%
def embed_data(embedder, loader, degree, adj, features, hashedFeatures, graph):
    dgraph = dgl.from_networkx(nx.Graph(graph)).to(device)
    features = features.cuda(device)
    if (embedder == 'hash'):
        embeds = hashedFeatures
        embeds_np = embeds.cpu().detach().numpy()
        loader.storeEmb('datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'a'+'_'+hashType+'.csv',embeds_np)
    elif (embedder == 'GNN'):
        featuresPlusHashed = torch.cat((features,hashedFeatures),1)
        embeds = model.gnn(dgraph,featuresPlusHashed)
        embeds_np = embeds.cpu().detach().numpy()
        loader.storeEmb('datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'b'+'_'+hashType+'.csv',embeds_np)
    else:
        featuresPlusHashed = torch.cat((features,hashedFeatures), 1)
        embeds2 = model.gnn(dgraph,featuresPlusHashed)
        embeds = torch.cat((hashedFeatures,embeds2),1)
        embeds_np = embeds.cpu().detach().numpy()
        loader.storeEmb('datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'c'+'_'+hashType+'.csv',embeds_np)
# %% [markdown]
# # Classify

# %%
def classify(data, classifierer, activation2, activationMLP, activationTrain): 
    from sklearn.model_selection import StratifiedKFold, train_test_split
    split=10
    skf=StratifiedKFold(n_splits=split, random_state=None, shuffle=True)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import ExtraTreesClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report,confusion_matrix,f1_score,accuracy_score,roc_auc_score

    global xgb_results_string
    global et_results_string
    encoder = '_hash-'+NNType
    
    print('Dataset: ' + data, file=logfile)
    print('Encoder: ' + encoder, file=logfile)
    print(notes, file=logfile)
    
    ### Load Data    
    X_all=pd.read_csv('datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'a'+'_'+hashType+'.csv', sep= ',', header=None,index_col=0).sort_index().values
    print('Reading file: ','datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'a'+'_'+hashType+'.csv')
    print('Reading file: ','datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'a'+'_'+hashType+'.csv', file = logfile)
    y_all=pd.read_csv('datasets/'+ data+'/data/'+data+'_labels.csv', sep= ',', header=None,index_col=0).sort_index().values.flatten()
    print('Reading file: ', 'datasets/'+ data+'/data/'+data+'_labels.csv')
    print('Reading file: ', 'datasets/'+ data+'/data/'+data+'_labels.csv', file = logfile)
    
    if 'tune' in mode or 'test' in mode: #use subset of data
        # Split out test data for separate use
        # Default to use tune set
        if 'tune' in mode:
            print('Classifier using tune subset data')
            print('Classifier using tune subset data', file=logfile)
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size=.2, random_state=10111221)
        if 'test' in mode: # use only test set
            X_train = X_test
            y_train = y_test
            print('Classifier using test subset data')
            print('Classifier using test subset data', file=logfile)     
    elif 'all' in mode: #use all data
        X_train = X_all
        y_train = y_all
        print('Classifier using all data')
        print('Classifier using all data', file=logfile)

    ### Classifier 1
    if (classifierer == 'xgboost' or classifierer =='all'):
        f1_values=0
        acc_values=0
        auroc_values=0
        
        for traintrain_index, val_index in skf.split(X_train, y_train):
            X_traintrain, X_val = X_train[traintrain_index], X_train[val_index]
            y_traintrain, y_val = y_train[traintrain_index], y_train[val_index]
            clf = XGBClassifier()
            clf.fit(X_traintrain, y_traintrain)
            # y_pred0=clf.predict(X_traintrain)
            y_pred=clf.predict(X_val)
            y_pred_proba = clf.predict_proba(X_val)
            f1_values+=f1_score(y_val, y_pred, average='weighted')
            # print('F1: ', f1_score(y_val, y_pred, average='weighted'))
            acc_values+=accuracy_score(y_val, y_pred)
            # print('Acc: ', accuracy_score(y_val, y_pred))
            # auroc_values+=roc_auc_score(y_val, y_pred_proba,multi_class='ovr')
            # print(classification_report(y_val, y_pred))
            # print(confusion_matrix(y_val, y_pred))
        print('average XGB Accuracy', acc_values/split)
        print('average XGB Accuracy', acc_values/split, file = logfile)
        print('average XGB F1', f1_values/split)
        print('average XGB F1', f1_values/split, file = logfile)
        # print('average XGB AUROC ' , auroc_values/split)

        # print(acc_values/split, ',', end='', file=resultsfileXGB)
        # print(f1_values/split, ',', end='', file=resultsfileXGB)
        xgb_results_string += str(acc_values/split)+','+str(f1_values/split)+','

    if (classifierer == 'ExtraTrees' or classifierer =='all'):
        f1_values=0
        acc_values=0
        auroc_values=0
        
        for traintrain_index, val_index in skf.split(X_train, y_train):
            X_traintrain, X_val = X_train[traintrain_index], X_train[val_index]
            y_traintrain, y_val = y_train[traintrain_index], y_train[val_index]
            clf =ExtraTreesClassifier(max_depth=30,n_estimators=100)
            clf.fit(X_traintrain, y_traintrain)
            # y_pred0=clf.predict(X_traintrain)
            y_pred=clf.predict(X_val)
            # y_pred_proba = clf.predict_proba(X_val)
            f1_values+=f1_score(y_val, y_pred, average='weighted')
            acc_values+=accuracy_score(y_val, y_pred)
            # auroc_values+=roc_auc_score(y_val, y_pred_proba)
            # print(classification_report(y_val, y_pred))
            # print(confusion_matrix(y_val, y_pred))    
        print('average ET Accuracy', acc_values/split)
        print('average ET Accuracy', acc_values/split, file = logfile)
        print('average ET F1', f1_values/split)
        print('average ET F1', f1_values/split, file = logfile)
        # print('average AUROC ' , auroc_values/split)

        # print(acc_values/split, ',', end='', file = resultsfileET)
        # print(f1_values/split, ',', end='', file = resultsfileET)
        et_results_string += str(acc_values/split)+','+str(f1_values/split)+','
    
    
    ### Classifier 2
    encoder = '_hash_'+NNType+'b'
    
    # Load Embeddings    
    X_all=pd.read_csv('datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'b'+'_'+hashType+'.csv', sep= ',', header=None,index_col=0).sort_index().values
    print('Reading file: ','datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'b'+'_'+hashType+'.csv')
    print('Reading file: ','datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'b'+'_'+hashType+'.csv', file = logfile)

    if 'tune' in mode or 'test' in mode: #use subset of data
        # Split out test data for separate use
        # Default to use tune set
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size=.2, random_state=10111221)
        if 'test' in mode: # use only test set
            X_train = X_test
            y_train = Y_test        
    elif 'all' in mode: #use all data
        X_train = X_all
        y_train = y_all
    

    if (classifierer == 'xgboost' or classifierer =='all'):
        f1_values=0
        acc_values=0
        
        for traintrain_index, val_index in skf.split(X_train, y_train):
            X_traintrain, X_val = X_train[traintrain_index], X_train[val_index]
            y_traintrain, y_val = y_train[traintrain_index], y_train[val_index]
            clf = XGBClassifier()
            clf.fit(X_traintrain, y_traintrain)
            # y_pred0=clf.predict(X_traintrain)
            y_pred=clf.predict(X_val)
            y_pred_proba = clf.predict_proba(X_val)
            f1_values+=f1_score(y_val, y_pred, average='weighted')
            # print('F1: ', f1_score(y_val, y_pred, average='weighted'))
            acc_values+=accuracy_score(y_val, y_pred)
            # print('Acc: ', accuracy_score(y_val, y_pred))
            # print(classification_report(y_val, y_pred))
            # print(confusion_matrix(y_val, y_pred))
        print('average XGB Accuracy', acc_values/split)
        print('average XGB Accuracy', acc_values/split, file = logfile)
        print('average XGB F1', f1_values/split)
        print('average XGB F1', f1_values/split, file = logfile)

        # print(acc_values/split, ',', end='', file = resultsfileXGB)
        # print(f1_values/split, ',', end='', file = resultsfileXGB)
        xgb_results_string += str(acc_values/split)+','+str(f1_values/split)+','

    if (classifierer == 'ExtraTrees' or classifierer =='all'):
        f1_values=0
        acc_values=0
        
        for traintrain_index, val_index in skf.split(X_train, y_train):
            X_traintrain, X_val = X_train[traintrain_index], X_train[val_index]
            y_traintrain, y_val = y_train[traintrain_index], y_train[val_index]
            clf =ExtraTreesClassifier(max_depth=30,n_estimators=100)
            clf.fit(X_traintrain, y_traintrain)
            # y_pred0=clf.predict(X_traintrain)
            y_pred=clf.predict(X_val)
            # y_pred_proba = clf.predict_proba(X_val)
            f1_values+=f1_score(y_val, y_pred, average='weighted')
            acc_values+=accuracy_score(y_val, y_pred)
            # print(classification_report(y_val, y_pred))
            # print(confusion_matrix(y_val, y_pred))    
        print('average ET Accuracy', acc_values/split)
        print('average ET Accuracy', acc_values/split, file = logfile)
        print('average ET F1', f1_values/split)
        print('average ET F1', f1_values/split, file = logfile)

        # print(acc_values/split, ',', end='', file = resultsfileET)
        # print(f1_values/split, ',', end='', file = resultsfileET)
        et_results_string += str(acc_values/split)+','+str(f1_values/split)+','
    
    ### Classifier 3
    encoder = '_hash-'+NNType+'c'
    
    # Load Embeddings
    X_all=pd.read_csv('datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'c'+'_'+hashType+'.csv', sep= ',', header=None,index_col=0).sort_index().values
    print('Reading file: ','datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'c'+'_'+hashType+'.csv')
    print('Reading file: ','datasets/'+ data+'/embeddings/emb_'+data+'_hash-'+NNType+'c'+'_'+hashType+'.csv', file = logfile)
    
    if 'tune' in mode or 'test' in mode: #use subset of data
        # Split out test data for separate use
        # Default to use tune set
        X_train, X_test, y_train, y_test = train_test_split(X_all, y_all,test_size=.2, random_state=10111221)
        if 'test' in mode: # use only test set
            X_train = X_test
            y_train = Y_test        
    elif 'all' in mode: #use all data
        X_train = X_all
        y_train = y_all


    if (classifierer == 'xgboost' or classifierer =='all'):
        f1_values=0
        acc_values=0
        
        for traintrain_index, val_index in skf.split(X_train, y_train):
            X_traintrain, X_val = X_train[traintrain_index], X_train[val_index]
            y_traintrain, y_val = y_train[traintrain_index], y_train[val_index]
            clf = XGBClassifier()
            clf.fit(X_traintrain, y_traintrain)
            # y_pred0=clf.predict(X_traintrain)
            y_pred=clf.predict(X_val)
            y_pred_proba = clf.predict_proba(X_val)
            f1_values+=f1_score(y_val, y_pred, average='weighted')
            # print('F1: ', f1_score(y_val, y_pred, average='weighted'))
            acc_values+=accuracy_score(y_val, y_pred)
            # print('Acc: ', accuracy_score(y_val, y_pred))
            # print(classification_report(y_val, y_pred))
            # print(confusion_matrix(y_val, y_pred))
        print('average XGB Accuracy', acc_values/split)
        print('average XGB Accuracy', acc_values/split, file = logfile)
        print('average XGB F1', f1_values/split)
        print('average XGB F1', f1_values/split, file = logfile)

        # print(acc_values/split, ',', end='', file = resultsfileXGB)
        # print(f1_values/split, ',', end='', file = resultsfileXGB)
        xgb_results_string += str(acc_values/split)+','+str(f1_values/split)+','

    if (classifierer == 'ExtraTrees' or classifierer =='all'):
        f1_values=0
        acc_values=0
        
        for traintrain_index, val_index in skf.split(X_train, y_train):
            X_traintrain, X_val = X_train[traintrain_index], X_train[val_index]
            y_traintrain, y_val = y_train[traintrain_index], y_train[val_index]
            clf =ExtraTreesClassifier(max_depth=30,n_estimators=100)
            clf.fit(X_traintrain, y_traintrain)
            # y_pred0=clf.predict(X_traintrain)
            y_pred=clf.predict(X_val)
            # y_pred_proba = clf.predict_proba(X_val)
            f1_values+=f1_score(y_val, y_pred, average='weighted')
            acc_values+=accuracy_score(y_val, y_pred)
            # print(classification_report(y_val, y_pred))
            # print(confusion_matrix(y_val, y_pred))    
        print('average ET Accuracy', acc_values/split)
        print('average ET Accuracy', acc_values/split, file = logfile)
        print('average ET F1', f1_values/split)
        print('average ET F1', f1_values/split, file = logfile)

        # print(acc_values/split, ',', end='', file = resultsfileET)
        # print(f1_values/split, ',', end='', file = resultsfileET)
        et_results_string += str(acc_values/split)+','+str(f1_values/split)+','

# %% [markdown]
# ## Scriptify

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset', type=str, default="cornell")
    parser.add_argument('--device', type=int, default=1) #-1 for CPU
    parser.add_argument('--GNN', type=str, default='GIN')
    parser.add_argument('--classifier', type=str, default="all") # or xgboost or ExtraTrees
    parser.add_argument('--activation1', type=str, default=None)
    parser.add_argument('--activation2', type=str, default=None)
    parser.add_argument('--activationMLP', type=str, default=None)
    parser.add_argument('--activationTrain', type=str, default=None)
    parser.add_argument('--epochs', type=int, default = 100000)
    parser.add_argument('--numTraining', type=int, default=10)
    
    parser.add_argument('--hashtype', type=str, default='SHA256') # or 'SHA256' or 'SHA3'
    parser.add_argument('--noisenum', type=int, default = 0)
    parser.add_argument('--mu', type=float, default = 0.0)
    parser.add_argument('--sigma', type=int, default = 1.0)

    parser.add_argument('--mode', type=str, default='tune') # or tune or test
    parser.add_argument('--trainType', type=str, default='new') # new or load
    parser.add_argument('--hiddenDim', type=int, default=2000)
    parser.add_argument('--GINLayers', type=int, default = None)
    parser.add_argument('--GNNLayers', type=int, default = None)


    args = parser.parse_args()
    data = args.dataset
    notes = 'Notes: GNN layers use activation2, none still runs batchnorms'
    NNType = args.GNN
    classifierer = args.classifier    
    device = args.device
    activation2 = args.activation2
    activationMLP = args.activationMLP
    activationTrain = args.activationTrain

    if activation2 == None:
        if NNType == "GCN":
            activation1 = 'sigmoid'
            activation2 = 'sigmoid'
            activationMLP = ''
            activationTrain = 'sigmoid'
        else:
            activation1 = 'relu'
            activation2 = 'relu'
            activationMLP = ''
            activationTrain = 'relu'

    nb_epochs = args.epochs
    numTraining = args.numTraining
    
    hashType = args.hashtype # or 'SHA256' or 'SHA3'
    noiseNum = args.noisenum
    mu = args.mu
    sigma = args.sigma
    
    mode = args.mode  
    trainType = args.trainType
    hiddendimension = args.hiddenDim
    GNNLayers = args.GNNLayers
    if GNNLayers == None:
        if NNType == "GCN":
            GINLayers = 2
            GNNLayers = 4
        else:
            GINLayers = 2
            GNNLayers = 2

    if (device == -1):
        cudaIsTrue = False
    else:
        cudaIsTrue=torch.cuda.is_available()
        print(cudaIsTrue)
        if (cudaIsTrue):
            torch.cuda.set_device(device)
            print(torch.cuda.current_device())
    
      
    logfile = open('datasets/'+ data+'/logs/'+data+'_hash-'+NNType+'_'+mode+'.log', 'w')
    xgb_results_string = ''
    et_results_string = ''
    for i in range(numTraining):
        iteration = str(i)
        loadert, features, adj, G, graph, degree = load(data)
        print('Training iteration ', iteration, ' at ', datetime.now(), '----------------------------------------------------------------------------------------')
        print('Training iteration ', iteration, ' at ', datetime.now(), '----------------------------------------------------------------------------------------', file=logfile)
        model = do_training(data, features, degree, adj, G, graph, activation2, activationMLP, activationTrain, noiseNum, mu,sigma)
        store(data, model, loadert, degree, adj, features, graph, activation2, activationMLP, activationTrain)
        print('Classifying iteration ', iteration, ' at ', datetime.now(), '----------------------------------------------------------------------------------------')
        print('Classifying iteration ', iteration, ' at ', datetime.now(), '----------------------------------------------------------------------------------------', file=logfile)
        classify(data, classifierer, activation2, activationMLP, activationTrain)
    logfile.close()
    # setup results files
    if (classifierer == 'xgboost' or classifierer =='all'):
        if 'tune' in mode:
            resultsfileXGB = open('datasets/'+ data+'/'+data+'_hash-'+NNType+'_XGB_HASHTYPES.csv', 'a')
            print(hashType + ',', end = '', file=resultsfileXGB)
            print(xgb_results_string, file=resultsfileXGB)
            resultsfileXGB.close()
        elif 'test' in mode:
            resultsfileXGB = open('XGB_TEST_RESULTS.csv', 'a')
            print(data + ','+ 'hash-'+NNType+ ',', end = '', file=resultsfileXGB)
            print(xgb_results_string, file=resultsfileXGB)
            resultsfileXGB.close()
        elif 'all' in mode:
            resultsfileXGB = open('XGB_ALL_RESULTS.csv', 'a')
            print(data + ','+ 'hash-'+NNType+ ',', end = '', file=resultsfileXGB)
            print(xgb_results_string, file=resultsfileXGB)
            resultsfileXGB.close()
    if (classifierer == 'ExtraTrees' or classifierer =='all'):
        if 'tune' in mode:
            resultsfileET = open('datasets/'+ data+'/'+data+'_hash-'+NNType+'_ET_HASHTYPES.csv', 'a')
            print(hashType+ ',', end = '', file=resultsfileET)
            print(et_results_string, file=resultsfileET)
            resultsfileET.close()
        elif 'test' in mode:
            resultsfileET = open('ET_TEST_RESULTS.csv', 'a')
            print(data + ','+ 'hash-'+NNType+ ',', end = '', file=resultsfileET)
            print(et_results_string, file=resultsfileET)
            resultsfileET.close()
        elif 'all' in mode:
            resultsfileET = open('ET_ALL_RESULTS.csv', 'a')
            print(data + ','+ 'hash-'+NNType+ ',', end = '', file=resultsfileET)
            print(et_results_string, file=resultsfileET)
            resultsfileET.close()
    


