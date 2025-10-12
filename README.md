# DeepSurrogate_model


This Python package provides flexible building blocks to construct explainable surrogate models for complex, high-fidelity computer simulations.

The core idea follows the architecture described in:

**DeepSurrogate: An Interpretable Artificial Intelligence System for Efficient Modeling of Functional Surrogates for High-Fidelity Computer Models**

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

```bash
python - <<'PY'
import numpy as np
from sklearn.model_selection import train_test_split
from DeepSurrogate_model.models import get_model_deepsurrogate
import tensorflow as tf

# -----------------------------
# 1) Toy data: y (s,m), x (s,p), z (m,q)
# -----------------------------
s, m, p, q = 300, 148, 2, 5
spatial_dim = 2
rng = np.random.default_rng(42)

coords = rng.uniform(-1, 1, size=(m, spatial_dim)).astype("float32")  # (m, spatial_dim)
x = rng.normal(size=(s, p)).astype("float32")                          # (s, p)

# z with repeated blocks along ROI axis: (m, q)
base = rng.normal(size=(q,)).astype("float32")
rep = 10
tmpl = np.repeat(base, rep)
z_full = np.resize(tmpl, m)  # cycle to length m
z = np.stack([np.roll(z_full, k) for k in range(q)], axis=1).astype("float32")  # (m, q)

# Generate y: (s, m)
beta_g = rng.normal(size=p)
beta_l = rng.normal(size=q)
spat = (coords @ rng.normal(size=(spatial_dim,1))).squeeze()
spat = (spat - spat.mean())/(spat.std()+1e-8)
y = np.zeros((s, m), dtype="float32")
for i in range(s):
    mu = 0.5*spat + x[i]@beta_g + z@beta_l
    y[i] = mu + rng.normal(scale=0.3, size=m)

# -----------------------------
# 2) Pack to sample-wise tensors for Keras
#    inputs: [global_inp, spatial, local], target: (s*m,1)
# -----------------------------
def pack(y, x, z, coords):
    s, m = y.shape
    G = np.repeat(x, m, axis=0)                        # (s*m, p)
    S = np.tile(coords, (s,1))                         # (s*m, spatial_dim)
    L = np.tile(z, (s,1))                              # (s*m, q)
    Y = y.reshape(-1, 1)                               # (s*m, 1)
    return [G.astype("float32"), S.astype("float32"), L.astype("float32")], Y.astype("float32")

idx = np.arange(s)
rng.shuffle(idx)
tr, te = idx[:int(0.8*s)], idx[int(0.8*s):]
va_split = int(0.9*s)

X_tr, y_tr = pack(y[tr], x[tr], z, coords)
X_va, y_va = pack(y[idx[int(0.8*s):va_split]], x[idx[int(0.8*s):va_split]], z, coords)
X_te, y_te = pack(y[te], x[te], z, coords)

# -----------------------------
# 3) Build, train, evaluate
# -----------------------------
model = get_model_deepsurrogate(
    global_dim=p, spatial_dim=spatial_dim, local_dim=q,
    global_hidden=[16,8], spatial_hidden=[16,8],
    dropout_p=0.1, mc=True, final_act="linear"  # regression
)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

history = model.fit(X_tr, y_tr,
                    validation_data=(X_va, y_va),
                    epochs=10, batch_size=128, verbose=1)

# MC-dropout predictions on test
n_samples = 100
mc_preds = []
for _ in range(n_samples):
    mc_preds.append(model.predict(X_te, verbose=0).squeeze())
mc_preds = np.stack(mc_preds, axis=0)                  # (n_samples, N)
mean_pred = mc_preds.mean(axis=0)
rmse = float(np.sqrt(np.mean((y_te.squeeze() - mean_pred)**2)))

lower = np.percentile(mc_preds, 2.5, axis=0)
upper = np.percentile(mc_preds, 97.5, axis=0)
covered = (lower <= y_te.squeeze()) & (y_te.squeeze() <= upper)
coverage = float(covered.mean())
width = float((upper - lower).mean())

print(f"RMSE: {rmse:.4f}")
print(f"Coverage (95%): {coverage:.4f}")
print(f"Mean interval width: {width:.4f}")
PY
```
