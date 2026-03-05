import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import lightgbm as lgb
from sklearn.multioutput import MultiOutputClassifier
from feature_engineering import build_behavior_features

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 256
EPOCHS = 20
LR = 2e-4
WEIGHT_DECAY = 5e-4

EMBED_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 6
DROPOUT = 0.3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "dataset")

# ======================
# LOAD DATA
# ======================

X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
Y_train = pd.read_csv(os.path.join(DATA_DIR, "Y_train.csv"))

X_val = pd.read_csv(os.path.join(DATA_DIR, "X_val.csv"))
Y_val = pd.read_csv(os.path.join(DATA_DIR, "Y_val.csv"))

feature_cols = [c for c in X_train.columns if c.startswith("feature_")]
SEQ_LEN = len(feature_cols)

# ======================
# VOCAB
# ======================

all_actions = pd.concat([X_train[feature_cols], X_val[feature_cols]]).values.flatten()
unique_actions = np.unique(all_actions)
unique_actions = unique_actions[unique_actions != 0]

action2idx = {a: i+2 for i,a in enumerate(unique_actions)}
action2idx[0] = 0
action2idx["UNK"] = 1

VOCAB_SIZE = len(action2idx)

def prepare_sequences(df):

    seq = df[feature_cols].values
    mapped = np.zeros_like(seq)

    for i in range(seq.shape[0]):
        for j in range(seq.shape[1]):
            mapped[i,j] = action2idx.get(seq[i,j],1)

    return torch.tensor(mapped,dtype=torch.long)

X_tr_seq = prepare_sequences(X_train)
X_va_seq = prepare_sequences(X_val)

# ======================
# LABELS
# ======================

label_maps = {}
num_classes = []

for col in Y_train.columns[1:]:

    uniq = sorted(Y_train[col].unique())

    mapping = {v:i for i,v in enumerate(uniq)}

    label_maps[col] = mapping

    num_classes.append(len(uniq))

    Y_train[col] = Y_train[col].map(mapping)
    Y_val[col] = Y_val[col].map(mapping)

y_tr = torch.tensor(Y_train.drop(columns=["id"]).values)
y_va = torch.tensor(Y_val.drop(columns=["id"]).values)

# ======================
# DATASET
# ======================

class SeqDataset(Dataset):

    def __init__(self,X,y):

        self.X = X
        self.y = y

    def __len__(self):

        return len(self.X)

    def __getitem__(self,idx):

        return self.X[idx],self.y[idx]


train_loader = DataLoader(
    SeqDataset(X_tr_seq,y_tr),
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    SeqDataset(X_va_seq,y_va),
    batch_size=BATCH_SIZE
)

# ======================
# POSITIONAL ENCODING
# ======================

class PositionalEncoding(nn.Module):

    def __init__(self,d_model,max_len):

        super().__init__()

        pe = torch.zeros(max_len,d_model)

        position = torch.arange(0,max_len).unsqueeze(1)

        div_term = torch.exp(torch.arange(0,d_model,2) * (-math.log(10000)/d_model))

        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        self.pe = pe.unsqueeze(0)

    def forward(self,x):

        return x + self.pe[:,:x.size(1)].to(x.device)

# ======================
# MODEL
# ======================

class TransformerModel(nn.Module):

    def __init__(self,vocab_size,num_classes):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size,EMBED_DIM,padding_idx=0)

        self.cls_token = nn.Parameter(torch.randn(1,1,EMBED_DIM))

        self.pos_encoder = PositionalEncoding(EMBED_DIM,SEQ_LEN+1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM*4,
            dropout=DROPOUT,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.heads = nn.ModuleList([
            nn.Linear(EMBED_DIM,n)
            for n in num_classes
        ])

    def forward(self,x,return_embedding=False):

        batch = x.size(0)

        emb = self.embedding(x)

        cls = self.cls_token.expand(batch,-1,-1)

        emb = torch.cat([cls,emb],dim=1)

        mask = torch.cat([
            torch.zeros((batch,1),dtype=torch.bool,device=x.device),
            (x==0)
        ],dim=1)

        emb = self.pos_encoder(emb)

        out = self.transformer(emb,src_key_padding_mask=mask)

        pooled = out[:,0]

        outputs = [head(pooled) for head in self.heads]

        if return_embedding:

            return outputs,pooled

        return outputs

model = TransformerModel(VOCAB_SIZE,num_classes).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(),lr=LR)

criterion = nn.CrossEntropyLoss()

# ======================
# TRAIN
# ======================

for epoch in range(EPOCHS):

    model.train()

    for xb,yb in tqdm(train_loader):

        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(xb)

        loss = sum(
            criterion(outputs[i],yb[:,i])
            for i in range(6)
        )

        loss.backward()

        optimizer.step()

# ======================
# EMBEDDINGS
# ======================

def extract_embeddings(model,loader):

    model.eval()

    embs=[]

    with torch.no_grad():

        for xb,yb in loader:

            xb = xb.to(DEVICE)

            _,pooled = model(xb,return_embedding=True)

            embs.append(pooled.cpu().numpy())

    return np.vstack(embs)

train_emb = extract_embeddings(model,train_loader)
val_emb = extract_embeddings(model,val_loader)

# ======================
# FEATURE ENGINEERING
# ======================

train_feat = build_behavior_features(X_train)
val_feat = build_behavior_features(X_val)

X_train_final = np.hstack([train_emb,train_feat])
X_val_final = np.hstack([val_emb,val_feat])

# ======================
# LIGHTGBM
# ======================

params = dict(

    n_estimators=1200,
    learning_rate=0.01,
    num_leaves=128,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.8,
)

lgb_model = MultiOutputClassifier(
    lgb.LGBMClassifier(**params)
)

lgb_model.fit(X_train_final,y_tr.numpy())

pred = lgb_model.predict(X_val_final)

exact = (pred == y_va.numpy()).all(axis=1).mean()

print("Validation Exact:",exact)

# ======================
# TEST
# ======================

X_test = pd.read_csv(os.path.join(DATA_DIR,"X_test.csv"))

X_test_seq = prepare_sequences(X_test)

test_loader = DataLoader(
    SeqDataset(X_test_seq,torch.zeros(len(X_test_seq),6)),
    batch_size=BATCH_SIZE
)

test_emb = extract_embeddings(model,test_loader)

test_feat = build_behavior_features(X_test)

X_test_final = np.hstack([test_emb,test_feat])

pred = lgb_model.predict(X_test_final)

# reverse labels

for i,col in enumerate(Y_train.columns[1:]):

    reverse = {v:k for k,v in label_maps[col].items()}

    pred[:,i] = np.vectorize(reverse.get)(pred[:,i])

pred = pred.astype(np.uint16)

submission = pd.DataFrame(pred,columns=Y_train.columns[1:])
submission.insert(0,"id",X_test["id"])

submission.to_csv("submission.csv",index=False)

print("submission saved")