# Code


This Python package provides flexible building blocks to construct explainable surrogate models for complex, high-fidelity computer simulations.

The core idea follows the architecture described in:


---

## Overview

This package implements:
- Global basis covariate blocks
- Spatial coordinates and local covariates
- Flexible functional surrogate modeling
- Monte Carlo dropout for quantifying uncertainty

---

## Installation

```bash
# Clone the repository
git clone https://github.com/jeon9677/DeepSurrogate_model.git

# Change into the directory
cd DeepSurrogate_model

# Install locally
pip install .

```
---

## A basic example of using the package
Note: The toy example below only illustrates the data structure used to run the model. It does not replicate the full complexity of the simulations used in the paper. For actual analysis, please follow the simulation design described in the paper. The final element of `global_hidden` and `spatial_hidden` in the `get_model_deepsurrogate` function must be equal, since the global and spatial branch outputs are combined via element-wise multiplication (Multiply layer).

```python

import argparse, numpy as np, tensorflow as tf
from DeepSurrogate_model import train_model, mc_predict
```


def make_data(s=300, m=148, p=2, q=5, spatial_dim=2, seed=42):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(-1, 1, size=(m, spatial_dim)).astype("float32")
    x = rng.normal(size=(s, p)).astype("float32")
    base = rng.normal(size=(q,)).astype("float32")
    rep = 10
    tmpl = np.repeat(base, rep)
    z_full = np.resize(tmpl, m)
    z = np.stack([np.roll(z_full, k) for k in range(q)], axis=1).astype("float32")
    beta_g = rng.normal(size=p); beta_l = rng.normal(size=q)
    spat = (coords @ rng.normal(size=(spatial_dim,1))).squeeze()
    spat = (spat - spat.mean())/(spat.std()+1e-8)
    y = np.stack([0.5*spat + x[i]@beta_g + z@beta_l +
                  rng.normal(scale=0.3, size=m) for i in range(s)]).astype("float32")
    return y, x, z, coords

def pack(y, x, z, coords):
    s, m = y.shape; p = x.shape[1]; q = z.shape[1]; sd = coords.shape[1]
    G = np.repeat(x, m, axis=0).reshape(s*m, p)
    S = np.tile(coords, (s,1)).reshape(s*m, sd)
    L = np.tile(z, (s,1)).reshape(s*m, q)
    Y = y.reshape(s*m, 1)
    return [G.astype("float32"), S.astype("float32"), L.astype("float32")], Y.astype("float32")

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--s", type=int, default=300)
    ap.add_argument("--m", type=int, default=148)
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--spatial-dim", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--mc", type=lambda s: s.lower() in {"1","true","yes","y"}, default=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    y, x, z, coords = make_data(args.s, args.m, args.p, args.q, args.spatial_dim, args.seed)
    idx = np.arange(args.s); np.random.default_rng(args.seed).shuffle(idx)
    tr, va, te = idx[:int(0.7*args.s)], idx[int(0.7*args.s):int(0.85*args.s)], idx[int(0.85*args.s):]
    X_tr, y_tr = pack(y[tr], x[tr], z, coords)
    X_va, y_va = pack(y[va], x[va], z, coords)
    X_te, y_te = pack(y[te], x[te], z, coords)

    # model = get_model_deepsurrogate(global_dim=args.p, spatial_dim=args.spatial_dim,
                                    local_dim=args.q, mc=args.mc, final_act="linear")

    # model is already compiled inside get_model_deepsurrogate
    # (Adam optimizer with ExponentialDecay learning rate schedule, gaussian_nll loss)
    # model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=args.epochs, batch_size=args.batch_size, verbose=1)

    model, history = train_model(
    X_tr, y_tr, X_va, y_va,
    global_dim=args.p, spatial_dim=args.spatial_dim, local_dim=args.q,
    mc=args.mc, final_act="linear",
    epochs=args.epochs, batch_size=args.batch_size)
    
    # MC dropout prediction (epistemic + aleatoric uncertainty)
    mean_pred, epistemic_var, aleatoric_var, total_var = mc_predict(model, X_te, n_samples=100)
    rmse = float(np.sqrt(np.mean((y_te.squeeze() - mean_pred)**2)))
    print(f"RMSE: {rmse:.4f}")
    print(f"Mean predictive std: {np.sqrt(total_var).mean():.4f}")

if __name__ == "__main__":
    main()

```
