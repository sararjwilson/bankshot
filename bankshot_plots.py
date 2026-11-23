import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

torch.manual_seed(0)
np.random.seed(0)

# raw data observations
df = pd.read_csv("shots.csv")
print(f"{len(df):,} bank shots, make rate {df.made.mean():.1%}")
df.head()

df["impact_speed"] = np.hypot(df.impact_vx, df.impact_vy)
df["impact_angle"] = np.degrees(np.arctan2(-df.impact_vy, df.impact_vx))
df[["impact_y", "impact_speed", "impact_angle", "impact_spin", "made"]].describe().round(2)

# make rate vs 1st impact height
bins = np.arange(3.10, 4.00, 0.05)
df["y_bin"] = pd.cut(df.impact_y, bins)
rate_y = df.groupby("y_bin", observed=True).made.mean()

# make rate vs impact angle/shot steepness
bins = np.arange(0, 61, 3)
df["angle_bin"] = pd.cut(df.impact_angle, bins)
rate_angle = df.groupby("angle_bin", observed=True).made.mean()

# build the learning model
# tensor-ize data
FEATURES_ALL = ["impact_y", "impact_speed", "impact_angle", "impact_spin"]

n = len(df)
perm = np.random.permutation(n)
i_train, i_val, i_test = np.split(perm, [int(0.7 * n), int(0.85 * n)])

def make_tensors(features):
    X = df[features].to_numpy(np.float32)
    y = df["made"].to_numpy(np.float32)
    mu, sd = X[i_train].mean(0), X[i_train].std(0)
    X = (X - mu) / sd
    to_t = lambda idx: (torch.tensor(X[idx]), torch.tensor(y[idx]).unsqueeze(1))
    return to_t(i_train), to_t(i_val), to_t(i_test), (mu, sd)

(Xtr, ytr), (Xva, yva), (Xte, yte), (mu, sd) = make_tensors(["impact_y"])
print(Xtr.shape, ytr.shape)   # (N, n_features), (N, 1)

# training loop
def accuracy(model, X, y):
    with torch.no_grad():
        return ((model(X) > 0).float() == y).float().mean().item()

def train_model(model, Xtr, ytr, Xva, yva, epochs=15, lr=1e-2, batch_size=1024):
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr))          # shuffle every epoch
        for start in range(0, len(Xtr), batch_size):
            idx = order[start:start + batch_size]
            xb, yb = Xtr[idx], ytr[idx]

            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            history["train_loss"].append(loss_fn(model(Xtr), ytr).item())
            history["val_loss"].append(loss_fn(model(Xva), yva).item())
        history["val_acc"].append(accuracy(model, Xva, yva))

    print(f"final val loss {history['val_loss'][-1]:.3f}, "
          f"val accuracy {history['val_acc'][-1]:.1%}")
    return history

# NN setup
class MLP(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x)

mlp_y = MLP(1)
hist_mlp_y = train_model(mlp_y, Xtr, ytr, Xva, yva)

# plots
# height
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([b.mid for b in rate_y.index], rate_y.values, "o-")
ax.axvline(3.048, color="orange", ls="--", label="rim height")
ax.set_xlabel("impact height of ball center (m)")
ax.set_ylabel("make rate")
ax.set_title("Height of First Impact with Backboard")
ax.legend();
# angle
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([b.mid for b in rate_angle.index], rate_angle.values, "o-")
ax.set_xlabel("impact angle of impact (deg)")
ax.set_ylabel("make rate")
ax.set_title("Steepness of Shot");

ep = range(1, len(hist_mlp_y["train_loss"]) + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.plot(ep, hist_mlp_y["train_loss"], label="train")
ax1.plot(ep, hist_mlp_y["val_loss"], label="val")
ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("Loss")
ax1.legend()
ax2.plot(ep, hist_mlp_y["val_acc"], color="C2")
ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Validation accuracy")
fig.tight_layout();

h = np.linspace(df.impact_y.min(), df.impact_y.max(), 300).astype(np.float32)
x = torch.tensor((h - mu) / sd).unsqueeze(1)

mlp_y.eval()
with torch.no_grad():
    p = torch.sigmoid(mlp_y(x)).squeeze(1).numpy()

fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot([b.mid for b in rate_y.index], rate_y.values, "o", label="observed (binned)")
ax.plot(h, p, "-", lw=2, label="MLP prediction")
ax.axvline(3.048, color="orange", ls="--", label="rim height")
ax.set_xlabel("impact height of ball center (m)")
ax.set_ylabel("P(make)")
ax.set_title("What the network learned")
ax.legend();
plt.show()