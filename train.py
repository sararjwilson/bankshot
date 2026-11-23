"""
train make/miss predictors on the simmed bank shots.

    python generate_data.py      # makes shots.csv if u dont have it
    python train.py              # prints test acc + saves plots to figures/

models
    baseline      just guess whatever happens more
    logistic      logistic regression, all impact features
    mlp_height    NN that only gets impact height
    mlp_all       NN with all impact features
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

FEATURES_HEIGHT = ["impact_y"]
FEATURES_ALL = ["impact_y", "impact_speed", "impact_angle", "impact_spin"]
RIM_Y = 3.048


# raw data

def load_shots(path):
    df = pd.read_csv(path)
    df["impact_speed"] = np.hypot(df.impact_vx, df.impact_vy)
    df["impact_angle"] = np.degrees(np.arctan2(-df.impact_vy, df.impact_vx))
    return df


def split_indices(n, seed):
    # 70/15/15 train/val/test
    perm = np.random.default_rng(seed).permutation(n)
    return np.split(perm, [int(0.7 * n), int(0.85 * n)])


def make_tensors(df, features, splits):
    # tensor-ize data, normalize w train stats only
    i_train = splits[0]
    X = df[features].to_numpy(np.float32)
    y = df["made"].to_numpy(np.float32)
    mu, sd = X[i_train].mean(0), X[i_train].std(0)
    X = (X - mu) / sd
    tensors = [(torch.tensor(X[i]), torch.tensor(y[i]).unsqueeze(1)) for i in splits]
    return tensors, (mu, sd)


# NN setup

class MLP(nn.Module):
    def __init__(self, n_in, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# training loop

def accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((model(X) > 0).float() == y).float().mean().item()


def train_model(model, train, val, epochs=15, lr=1e-2, batch_size=1024):
    (Xtr, ytr), (Xva, yva) = train, val
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr))          # shuffle every epoch
        for start in range(0, len(Xtr), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            history["train_loss"].append(loss_fn(model(Xtr), ytr).item())
            history["val_loss"].append(loss_fn(model(Xva), yva).item())
        history["val_acc"].append(accuracy(model, Xva, yva))
    return history


# plots

# make rate per bin
def binned_rate(df, col, bins):
    rate = df.groupby(pd.cut(df[col], bins), observed=True).made.mean()
    return np.array([b.mid for b in rate.index]), rate.values


def plot_make_rate(df, figdir):
    h, rate_h = binned_rate(df, "impact_y", np.arange(3.10, 4.00, 0.05))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(h, rate_h, "o-")
    ax.axvline(RIM_Y, color="orange", ls="--", label="rim height")
    ax.set_xlabel("impact height of ball center (m)")
    ax.set_ylabel("make rate")
    ax.set_title("Height of First Impact with Backboard")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "make_rate_vs_height.png"), dpi=150)
    plt.close(fig)

    a, rate_a = binned_rate(df, "impact_angle", np.arange(0, 61, 3))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(a, rate_a, "o-")
    ax.set_xlabel("impact angle (deg)")
    ax.set_ylabel("make rate")
    ax.set_title("Steepness of Shot")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "make_rate_vs_angle.png"), dpi=150)
    plt.close(fig)


def plot_training(history, figdir):
    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
    ax1.plot(ep, history["train_loss"], label="train")
    ax1.plot(ep, history["val_loss"], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("BCE loss"); ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(ep, history["val_acc"], color="C2")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("accuracy"); ax2.set_title("Validation accuracy")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "training_curves.png"), dpi=150)
    plt.close(fig)


def plot_learned_curve(df, model, mu, sd, figdir):
    h_obs, rate = binned_rate(df, "impact_y", np.arange(3.10, 4.00, 0.05))
    h = np.linspace(df.impact_y.min(), df.impact_y.max(), 300).astype(np.float32)
    x = torch.tensor((h - mu) / sd).unsqueeze(1)
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(x)).squeeze(1).numpy()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(h_obs, rate, "o", label="observed (binned)")
    ax.plot(h, p, "-", lw=2, label="MLP prediction")
    ax.axvline(RIM_Y, color="orange", ls="--", label="rim height")
    ax.set_xlabel("impact height of ball center (m)")
    ax.set_ylabel("P(make)")
    ax.set_title("What the network learned")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "learned_curve.png"), dpi=150)
    plt.close(fig)


# run everything

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="shots.csv")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.figdir, exist_ok=True)

    df = load_shots(args.data)
    print(f"{len(df):,} bank shots, make rate {df.made.mean():.1%}")
    splits = split_indices(len(df), args.seed)
    plot_make_rate(df, args.figdir)

    results = {}

    # baseline, always guess the more common outcome
    y = df["made"].to_numpy()
    majority = int(y[splits[0]].mean() >= 0.5)
    results["baseline (majority class)"] = (y[splits[2]] == majority).mean()

    # logistic regression w all features
    (tr, va, te), _ = make_tensors(df, FEATURES_ALL, splits)
    logistic = nn.Linear(len(FEATURES_ALL), 1)
    train_model(logistic, tr, va, epochs=args.epochs)
    results["logistic regression"] = accuracy(logistic, *te)

    # NN, height only
    (tr, va, te), (mu, sd) = make_tensors(df, FEATURES_HEIGHT, splits)
    mlp_height = MLP(1)
    train_model(mlp_height, tr, va, epochs=args.epochs)
    results["MLP, height only"] = accuracy(mlp_height, *te)
    plot_learned_curve(df, mlp_height, mu, sd, args.figdir)

    # NN, all features
    (tr, va, te), _ = make_tensors(df, FEATURES_ALL, splits)
    mlp_all = MLP(len(FEATURES_ALL))
    history = train_model(mlp_all, tr, va, epochs=args.epochs)
    results["MLP, full impact state"] = accuracy(mlp_all, *te)
    plot_training(history, args.figdir)

    print("\ntest accuracy")
    for name, acc in results.items():
        print(f"  {name:<26} {acc:.1%}")
    print(f"\nplots saved to {args.figdir}/")


if __name__ == "__main__":
    main()
