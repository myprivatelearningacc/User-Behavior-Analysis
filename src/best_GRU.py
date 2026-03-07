# ================================================================
# CELL 1: SETUP & INSTALL
# ================================================================


import os, warnings, math
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from gensim.models import Word2Vec

warnings.filterwarnings('ignore')


SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

ATTRS  = ['attr_1', 'attr_2', 'attr_3', 'attr_4', 'attr_5', 'attr_6']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

FOLDER = 'data' #← SỬA NẾU CẦN

TRAIN_X_PATH = FOLDER + 'X_train.csv'
TRAIN_Y_PATH = FOLDER + 'Y_train.csv'
VAL_X_PATH   = FOLDER + 'X_val.csv'
VAL_Y_PATH   = FOLDER + 'Y_val.csv'
TEST_X_PATH  = FOLDER + 'X_test.csv'
OUTPUT_PATH  = FOLDER + '../submission.csv'

for path in [TRAIN_X_PATH, TRAIN_Y_PATH, VAL_X_PATH, VAL_Y_PATH, TEST_X_PATH]:
    status = '✓' if os.path.exists(path) else '✗ MISSING'
    print(f"{status} {path}")

def parse_X_file(filepath):
    with open(filepath, 'r') as f:
        first_line = f.readline()
    delimiter = '\t' if '\t' in first_line else ','
    df = pd.read_csv(filepath, header=None, delimiter=delimiter, dtype=str)

    # Drop header row nếu có
    is_header = False
    for val in df.iloc[0].iloc[1:]:
        if pd.notna(val):
            try:
                float(val)
            except:
                is_header = True
                break
    if is_header:
        df = df.iloc[1:].reset_index(drop=True)

    sequences   = {}
    ids_ordered = []
    for _, row in df.iterrows():
        uid = str(row.iloc[0]).strip()
        actions = []
        for val in row.iloc[1:]:
            if pd.notna(val):
                try:
                    actions.append(int(float(val)))
                except:
                    pass
        sequences[uid] = actions
        ids_ordered.append(uid)
    return sequences, ids_ordered


print("Parsing sequences...")
train_seqs, train_ids = parse_X_file(TRAIN_X_PATH)
val_seqs,   val_ids   = parse_X_file(VAL_X_PATH)
test_seqs,  test_ids  = parse_X_file(TEST_X_PATH)

Y_train_raw = pd.read_csv(TRAIN_Y_PATH)
Y_val_raw   = pd.read_csv(VAL_Y_PATH)
ID_COL      = Y_train_raw.columns[0]

Y_train = Y_train_raw.set_index(ID_COL).loc[train_ids].reset_index()
Y_val   = Y_val_raw.set_index(ID_COL).loc[val_ids].reset_index()

assert list(Y_train[ID_COL]) == train_ids, "Train X/Y không match!"
assert list(Y_val[ID_COL])   == val_ids,   "Val X/Y không match!"

print(f"Train: {len(train_seqs):,} | Val: {len(val_seqs):,} | Test: {len(test_seqs):,}")

# Action frequency (dùng cho features)
all_actions_flat = [a for seq in train_seqs.values() for a in seq]
action_freq      = Counter(all_actions_flat)
print(f"Vocab size (unique actions in train): {len(action_freq):,}")
print("Data loaded ✓")

def extract_pos(seq):
    """Trích xuất positional features từ 1 sequence."""
    n = len(seq)
    def p(i):
        if i >= 0: return seq[i] if i < n      else -1
        else:      return seq[i] if abs(i) <= n else -1
    return {
        'first':       p(0),
        'second':      p(1),
        'third':       p(2),
        'pos4':        p(3),
        'pos5':        p(4),
        'pos_n4':      p(-4),
        'third_last':  p(-3),
        'second_last': p(-2),
        'last':        p(-1),
    }


# Best lookup keys — tìm được từ brute-force scan
LOOKUP_KEYS = {
    'attr_1': ['first', 'second', 'third', 'pos4'],
    'attr_2': ['second', 'third', 'pos4'],
    'attr_3': ['third', 'pos4', 'pos5', 'pos_n4', 'third_last', 'second_last', 'last'],
    'attr_4': ['third', 'pos4', 'pos5', 'pos_n4', 'third_last', 'second_last', 'last'],
    'attr_5': ['third', 'pos_n4', 'third_last', 'second_last', 'last'],
    'attr_6': ['pos_n4', 'third_last', 'second_last', 'last'],
}


def build_pos_df(seqs_dict, ids_list):
    rows = [extract_pos(seqs_dict[uid]) for uid in ids_list]
    return pd.DataFrame(rows, index=ids_list)


def build_lookup(pos_df, Y_df, keys_dict):
    """Chỉ giữ unambiguous entries (1 unique value per key combo)."""
    df = pos_df.copy()
    for attr in ATTRS:
        df[attr] = Y_df[attr].values
    tables = {}
    for attr, keys in keys_dict.items():
        g    = df.groupby(keys)[attr]
        hard = {kv: int(grp.mode()[0]) for kv, grp in g if grp.nunique() == 1}
        tables[attr] = hard
        print(f"  {attr}: {len(hard):,} unambiguous groups ({len(hard)/len(g)*100:.1f}%)")
    return tables


def apply_lookup(pos_df, lookup_tables, keys_dict):
    """Trả về predictions và mask (True = lookup hit)."""
    preds = {attr: np.full(len(pos_df), -1, dtype=int) for attr in ATTRS}
    masks = {attr: np.zeros(len(pos_df), dtype=bool)   for attr in ATTRS}
    for attr, keys in keys_dict.items():
        lk = lookup_tables[attr]
        for i, row in enumerate(pos_df[keys].itertuples(index=False)):
            kv = tuple(int(v) for v in row)
            if kv in lk:
                preds[attr][i] = lk[kv]
                masks[attr][i] = True
    return preds, masks


print("Building positional dataframes...")
pos_train = build_pos_df(train_seqs, train_ids)
pos_val   = build_pos_df(val_seqs,   val_ids)
pos_test  = build_pos_df(test_seqs,  test_ids)

print("\nBuilding lookup tables from TRAIN...")
lookup_tables = build_lookup(pos_train, Y_train, LOOKUP_KEYS)

# Evaluate lookup trên val
lk_val_pred, lk_val_mask = apply_lookup(pos_val, lookup_tables, LOOKUP_KEYS)
print("\nLookup coverage on VAL:")
for attr in ATTRS:
    cov     = lk_val_mask[attr].mean()
    hit_idx = np.where(lk_val_mask[attr])[0]
    acc     = np.mean(lk_val_pred[attr][hit_idx] == Y_val[attr].values[hit_idx]) \
              if len(hit_idx) else 0
    print(f"  {attr}: coverage={cov:.1%}, accuracy_when_hit={acc:.1%}")

print("\nBuilding vocabulary (Train + Val + Test)...")
all_action_ids = set()
for seqs_dict in [train_seqs, val_seqs, test_seqs]:
    for seq in seqs_dict.values():
        all_action_ids.update(seq)

action2idx          = {a: i + 2 for i, a in enumerate(sorted(all_action_ids))}
action2idx[0]       = 0   # padding
action2idx['UNK']   = 1   # unknown
VOCAB_SIZE          = len(action2idx) + 1

MAX_LEN = max(
    max(len(s) for s in train_seqs.values()),
    max(len(s) for s in val_seqs.values()),
    max(len(s) for s in test_seqs.values()),
)
print(f"Vocab size: {VOCAB_SIZE:,} | Max seq len: {MAX_LEN}")


def encode_and_pad(seqs_dict, ids_list, max_len=MAX_LEN):
    X = np.zeros((len(ids_list), max_len), dtype=np.int64)
    L = np.zeros(len(ids_list), dtype=np.int64)
    for i, uid in enumerate(ids_list):
        seq    = seqs_dict[uid]
        length = min(len(seq), max_len)
        for j in range(length):
            X[i, j] = action2idx.get(seq[j], 1)
        L[i] = max(length, 1)  # tránh length=0
    return torch.LongTensor(X), torch.LongTensor(L)


print("Encoding sequences...")
X_tr_seq, L_tr = encode_and_pad(train_seqs, train_ids)
X_va_seq, L_va = encode_and_pad(val_seqs,   val_ids)
X_te_seq, L_te = encode_and_pad(test_seqs,  test_ids)
print("Encoding done ✓")

# CELL 6: AUXILIARY FEATURES (Positional + Statistical)
# ================================================================
def build_aux(seqs_dict, ids_list, action_freq_ref):
    rows = []
    for uid in ids_list:
        seq = seqs_dict[uid]
        n   = len(seq)
        pos = extract_pos(seq)
        cnt = Counter(seq)

        # Entropy
        probs = np.array(list(cnt.values())) / n
        ent   = float(-np.sum(probs * np.log2(probs + 1e-10)))

        f = {
            # === Statistical ===
            'seq_len':       n,
            'n_unique':      len(set(seq)),
            'unique_ratio':  len(set(seq)) / n,
            'has_repeat':    int(n > len(set(seq))),
            'entropy':       ent,
            'mean_val':      float(np.mean(seq)),
            'std_val':       float(np.std(seq)),
            'max_val':       max(seq),
            'min_val':       min(seq),
            'median_val':    float(np.median(seq)),
            'most_common_freq': cnt.most_common(1)[0][1],

            # === Positional (SIGNAL CHÍNH) ===
            'first':         pos['first'],
            'second':        pos['second'],
            'third':         pos['third'],
            'pos4':          pos['pos4'],
            'pos5':          pos['pos5'],
            'pos_n4':        pos['pos_n4'],
            'third_last':    pos['third_last'],
            'second_last':   pos['second_last'],
            'last':          pos['last'],

            # === Global frequency ===
            'first_freq':    action_freq_ref.get(pos['first'],  0),
            'last_freq':     action_freq_ref.get(pos['last'],   0),
            'second_freq':   action_freq_ref.get(pos['second'], 0),

            # === 2-way interactions ===
            'f_x_s':    pos['first']  * 100000 + pos['second'],
            'f_x_t':    pos['first']  * 100000 + pos['third'],
            's_x_t':    pos['second'] * 100000 + pos['third'],
            't_x_p4':   pos['third']  * 100000 + pos['pos4'],
            'p4_x_p5':  pos['pos4']   * 100000 + pos['pos5'],
            'sl_x_l':   pos['second_last'] * 100000 + pos['last'],
            'tl_x_sl':  pos['third_last']  * 100000 + pos['second_last'],
            'pn4_x_tl': pos['pos_n4']      * 100000 + pos['third_last'],
            'f_x_l':    pos['first']  * 100000 + pos['last'],

            # === 3-way interactions (cho attrs khó) ===
            'f_s_t':    pos['first']  * 10**9 + pos['second'] * 10**4 + pos['third'],
            't_p4_p5':  pos['third']  * 10**9 + pos['pos4']   * 10**4 + pos['pos5'],
            'tl_sl_l':  pos['third_last'] * 10**9 + pos['second_last'] * 10**4 + pos['last'],
            'pn4_tl_sl':pos['pos_n4'] * 10**9 + pos['third_last'] * 10**4 + pos['second_last'],
        }
        rows.append(f)
    return pd.DataFrame(rows).fillna(-1)


print("Building auxiliary features...")
aux_tr_df = build_aux(train_seqs, train_ids, action_freq)
aux_va_df = build_aux(val_seqs,   val_ids,   action_freq)
aux_te_df = build_aux(test_seqs,  test_ids,  action_freq)

scaler  = StandardScaler()
aux_tr  = torch.FloatTensor(scaler.fit_transform(aux_tr_df))
aux_va  = torch.FloatTensor(scaler.transform(aux_va_df))
aux_te  = torch.FloatTensor(scaler.transform(aux_te_df))
AUX_DIM = aux_tr.shape[1]
print(f"Aux features: {AUX_DIM} dims ✓")


# ================================================================
# CELL 7: DATASET & DATALOADER
# ================================================================
N_CLASSES = {attr: int(Y_train[attr].max()) for attr in ATTRS}
print(f"N_CLASSES: {N_CLASSES}")

y_tr = torch.LongTensor(np.stack([Y_train[a].values - 1 for a in ATTRS], axis=1))
y_va = torch.LongTensor(np.stack([Y_val[a].values   - 1 for a in ATTRS], axis=1))


class SeqDataset(Dataset):
    def __init__(self, seq, lengths, aux, y=None):
        self.seq     = seq
        self.lengths = lengths
        self.aux     = aux
        self.y       = y

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.seq[idx], self.lengths[idx], self.aux[idx], self.y[idx]
        return self.seq[idx], self.lengths[idx], self.aux[idx]


BATCH_SIZE = 256

train_ds = SeqDataset(X_tr_seq, L_tr, aux_tr, y_tr)
val_ds   = SeqDataset(X_va_seq, L_va, aux_va, y_va)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print("Datasets ready ✓")


# ================================================================
# CELL 8: MODEL — ENHANCED BIDIRECTIONAL GRU + ATTENTION
# ================================================================
class EnhancedGRU(nn.Module):
    """
    Kiến trúc:
    - Bidirectional GRU (học cả forward + backward)
    - Attention pooling (focus vào positions quan trọng)
    - First + Last token embeddings (positional signal mạnh)
    - Auxiliary feature branch (positional + statistical)
    - 6 independent heads cho 6 attrs
    """
    def __init__(self, vocab_size, n_classes_dict, aux_dim,
                 embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim

        # Embedding layer
        self.embedding  = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_drop = nn.Dropout(0.1)

        # Bidirectional GRU
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        gru_out_dim = hidden_dim * 2  # bidirectional

        # Attention: học trọng số cho từng position
        self.attention = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        # Auxiliary feature branch
        self.aux_net = nn.Sequential(
            nn.Linear(aux_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
        )

        # Combined dim: attn_pool + first_emb + last_emb + aux
        combined_dim = gru_out_dim + embed_dim + embed_dim + 64

        # Per-attr heads
        self.heads = nn.ModuleDict({
            attr: nn.Sequential(
                nn.Linear(combined_dim, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(128, n_cls),
            )
            for attr, n_cls in n_classes_dict.items()
        })

    def forward(self, x, lengths, aux):
        B, T = x.shape

        # Embedding
        emb = self.embed_drop(self.embedding(x))  # (B, T, E)

        # Pack → GRU → Unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False,
        )
        gru_out_packed, _ = self.gru(packed)
        gru_out, _        = nn.utils.rnn.pad_packed_sequence(
            gru_out_packed, batch_first=True, total_length=T,
        )  # (B, T, H*2)

        # Attention pooling
        attn_scores  = self.attention(gru_out).squeeze(-1)  # (B, T)
        pad_mask     = torch.arange(T, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores  = attn_scores.masked_fill(pad_mask, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        attn_out     = (gru_out * attn_weights).sum(dim=1)              # (B, H*2)

        # First & Last token embeddings
        first_emb = emb[:, 0, :]
        last_idx  = (lengths - 1).clamp(min=0)
        last_emb  = emb[torch.arange(B, device=x.device), last_idx, :]

        # Auxiliary branch
        aux_feat = self.aux_net(aux)  # (B, 64)

        # Concatenate
        combined = torch.cat([attn_out, first_emb, last_emb, aux_feat], dim=1)

        return {attr: head(combined) for attr, head in self.heads.items()}


model = EnhancedGRU(
    vocab_size=VOCAB_SIZE,
    n_classes_dict=N_CLASSES,
    aux_dim=AUX_DIM,
).to(DEVICE)

n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")


# ================================================================
# CELL 9: TRAINING
# ================================================================
EPOCHS   = 60
LR       = 1e-3
PATIENCE = 8

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_exact   = 0.0
best_state   = None
patience_cnt = 0

print(f"\nTraining on {DEVICE}...")
print(f"{'Epoch':>6} | {'Loss':>8} | {'ExactMatch':>10} | {'Status'}")
print("-" * 55)

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss = 0.0
    for batch in train_dl:
        seq, lengths, aux_b, yb = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        outs = model(seq, lengths, aux_b)
        loss = sum(criterion(outs[a], yb[:, i]) for i, a in enumerate(ATTRS))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    scheduler.step()

    # --- Validate ---
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch in val_dl:
            seq, lengths, aux_b, yb = [b.to(DEVICE) for b in batch]
            outs  = model(seq, lengths, aux_b)
            preds = torch.stack([outs[a].argmax(1) for a in ATTRS], dim=1).cpu()
            all_preds.append(preds)
            all_true.append(yb.cpu())

    all_preds  = torch.cat(all_preds)
    all_true   = torch.cat(all_true)
    val_exact  = (all_preds == all_true).all(dim=1).float().mean().item()
    avg_loss   = total_loss / len(train_dl)

    is_best = val_exact > best_exact
    if is_best:
        best_exact   = val_exact
        best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience_cnt = 0
        status = f"✓ BEST ({best_exact:.4f})"
    else:
        patience_cnt += 1
        status = f"patience {patience_cnt}/{PATIENCE}"

    print(f"  {epoch+1:4d} | {avg_loss:8.4f} | {val_exact:10.4f} | {status}")

    if patience_cnt >= PATIENCE:
        print(f"\nEarly stopping at epoch {epoch+1}")
        break

print(f"\nBest Val Exact Match (GRU alone): {best_exact:.4f} ({best_exact:.1%})")

# Load best weights
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})


# ================================================================
# CELL 10: EVALUATE GRU TRÊN VAL
# ================================================================
model.eval()
gru_val_probs  = {attr: [] for attr in ATTRS}
gru_val_preds  = {attr: [] for attr in ATTRS}

with torch.no_grad():
    for batch in val_dl:
        seq, lengths, aux_b, _ = [b.to(DEVICE) for b in batch]
        outs = model(seq, lengths, aux_b)
        for attr in ATTRS:
            prob = torch.softmax(outs[attr], dim=1).cpu().numpy()
            pred = outs[attr].argmax(1).cpu().numpy() + 1
            gru_val_probs[attr].append(prob)
            gru_val_preds[attr].append(pred)

gru_val_probs = {attr: np.vstack(gru_val_probs[attr]) for attr in ATTRS}
gru_val_preds = {attr: np.concatenate(gru_val_preds[attr]) for attr in ATTRS}

print("Per-attr accuracy (GRU, val):")
for attr in ATTRS:
    acc = accuracy_score(Y_val[attr], gru_val_preds[attr])
    print(f"  {attr}: {acc:.1%}")

exact_gru = np.mean([
    all(gru_val_preds[a][i] == Y_val[a].iloc[i] for a in ATTRS)
    for i in range(len(Y_val))
])
print(f"\nGRU Val Exact Match: {exact_gru:.4f} ({exact_gru:.1%})")


# ================================================================
# CELL 11: GRU + LOOKUP OVERRIDE (val)
# ================================================================
print("\nApplying Lookup Override on VAL...")

final_val_preds = {attr: gru_val_preds[attr].copy() for attr in ATTRS}
lk_val_pred, lk_val_mask = apply_lookup(pos_val, lookup_tables, LOOKUP_KEYS)

override_val = 0
for attr in ATTRS:
    mask = lk_val_mask[attr]
    final_val_preds[attr][mask] = lk_val_pred[attr][mask]
    override_val += mask.sum()

exact_final_val = np.mean([
    all(final_val_preds[a][i] == Y_val[a].iloc[i] for a in ATTRS)
    for i in range(len(Y_val))
])

print(f"\n{'='*55}")
print(f"  GRU alone:              {exact_gru:.4f} ({exact_gru:.1%})")
print(f"  GRU + Lookup Override:  {exact_final_val:.4f} ({exact_final_val:.1%})")
print(f"  Lookup overrode {override_val} / {len(val_ids)} val predictions")
print(f"{'='*55}")

# Per-attr sau override
print("\nPer-attr accuracy AFTER override (val):")
for attr in ATTRS:
    acc_gru  = accuracy_score(Y_val[attr], gru_val_preds[attr])
    acc_final = accuracy_score(Y_val[attr], final_val_preds[attr])
    print(f"  {attr}: GRU={acc_gru:.1%} → After override={acc_final:.1%}")


# ================================================================
# CELL 12: RETRAIN TRÊN TRAIN+VAL
# ================================================================
print("\nRebuilding lookup from Train+Val combined...")
all_seqs = {**train_seqs, **val_seqs}
all_ids  = train_ids + val_ids
Y_all    = pd.concat([Y_train, Y_val], ignore_index=True)

pos_all = build_pos_df(all_seqs, all_ids)
print("Building lookup tables from Train+Val...")
lookup_all = build_lookup(pos_all, Y_all, LOOKUP_KEYS)

# Encode + Aux cho all
print("\nBuilding features for Train+Val combined...")
X_all_seq, L_all = encode_and_pad(all_seqs, all_ids)
aux_all_df        = build_aux(all_seqs, all_ids, action_freq)
scaler_full       = StandardScaler()
aux_all           = torch.FloatTensor(scaler_full.fit_transform(aux_all_df))
aux_te_final      = torch.FloatTensor(scaler_full.transform(aux_te_df))

y_all = torch.LongTensor(
    np.stack([Y_all[a].values - 1 for a in ATTRS], axis=1)
)

full_ds = SeqDataset(X_all_seq, L_all, aux_all, y_all)
full_dl = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Init model mới, load best weights làm starting point
model_full = EnhancedGRU(
    vocab_size=VOCAB_SIZE,
    n_classes_dict=N_CLASSES,
    aux_dim=AUX_DIM,
).to(DEVICE)
model_full.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

opt_full   = torch.optim.AdamW(model_full.parameters(), lr=LR * 0.3, weight_decay=1e-4)
sched_full = torch.optim.lr_scheduler.CosineAnnealingLR(opt_full, T_max=15)

print("\nFine-tuning on Train+Val (15 epochs)...")
for epoch in range(15):
    model_full.train()
    total_loss = 0.0
    for batch in full_dl:
        seq, lengths, aux_b, yb = [b.to(DEVICE) for b in batch]
        opt_full.zero_grad()
        outs = model_full(seq, lengths, aux_b)
        loss = sum(criterion(outs[a], yb[:, i]) for i, a in enumerate(ATTRS))
        loss.backward()
        nn.utils.clip_grad_norm_(model_full.parameters(), 1.0)
        opt_full.step()
        total_loss += loss.item()
    sched_full.step()
    print(f"  Epoch {epoch+1:2d}/15 | loss={total_loss/len(full_dl):.4f}")

print("Retrain done ✓")


# ================================================================
# CELL 13: PREDICT TEST SET & APPLY LOOKUP
# ================================================================
print("\nPredicting test set...")
te_ds = SeqDataset(X_te_seq, L_te, aux_te_final)
te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, num_workers=0)

model_full.eval()
test_preds = {attr: [] for attr in ATTRS}

with torch.no_grad():
    for batch in te_dl:
        seq, lengths, aux_b = [b.to(DEVICE) for b in batch]
        outs = model_full(seq, lengths, aux_b)
        for attr in ATTRS:
            pred = outs[attr].argmax(1).cpu().numpy() + 1
            test_preds[attr].append(pred)

test_preds = {attr: np.concatenate(test_preds[attr]) for attr in ATTRS}

# Apply lookup override trên test
print("Applying lookup override on test...")
pos_test_df2 = build_pos_df(test_seqs, test_ids)
lk_te_pred, lk_te_mask = apply_lookup(pos_test_df2, lookup_all, LOOKUP_KEYS)

override_test = 0
for attr in ATTRS:
    mask = lk_te_mask[attr]
    test_preds[attr][mask] = lk_te_pred[attr][mask]
    override_test += mask.sum()

print(f"Lookup overrode {override_test} / {len(test_ids)} test predictions "
      f"({override_test/len(test_ids):.1%})")


# ================================================================
# CELL 14: BUILD & SAVE SUBMISSION
# ================================================================
submission = pd.DataFrame({'id': test_ids})
for attr in ATTRS:
    submission[attr] = test_preds[attr].astype(np.uint16)

# Validate
print("\nValidation checks:")
assert len(submission) == len(test_ids),                       "❌ Row count!"
assert (submission[ATTRS] >= 1).all().all(),                   "❌ Value < 1!"
assert submission[['attr_1','attr_4']].max().max() <= 12,      "❌ attr_1/4 range!"
assert submission[['attr_2','attr_5']].max().max() <= 31,      "❌ attr_2/5 range!"
assert submission[['attr_3','attr_6']].max().max() <= 99,      "❌ attr_3/6 range!"
assert submission['attr_1'].dtype == np.uint16,                "❌ dtype!"
print("✓ All checks passed")

submission.to_csv(OUTPUT_PATH, index=False)
print(f"\nSubmission saved: {OUTPUT_PATH}")
print(f"Shape: {submission.shape}")
print(submission.head(10))

print(f"\n{'='*55}")
print(f"  SUMMARY")
print(f"{'='*55}")
print(f"  GRU alone (val):             {exact_gru:.4f} ({exact_gru:.1%})")
print(f"  GRU + Lookup (val):          {exact_final_val:.4f} ({exact_final_val:.1%})")
print(f"  Test lookup override:         {override_test}/{len(test_ids)} ({override_test/len(test_ids):.1%})")
print(f"  Submission:                   {OUTPUT_PATH}")
print(f"{'='*55}")