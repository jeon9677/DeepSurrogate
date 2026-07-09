import numpy as np
from .models import get_model_deepsurrogate

def train_model(
    X_train, y_train,
    X_val, y_val,
    global_dim,
    spatial_dim,
    local_dim,
    global_hidden=[16, 8],
    spatial_hidden=[16, 8],
    noise_hidden=[32, 16],
    dropout_p=0.1,
    mc=True,
    epochs=10,
    batch_size=128
):
    """
    Train the DeepSurrogate model.

    Args:
    X_train, X_val (list of np.ndarray): [global, spatial, local] input arrays
    y_train, y_val (np.ndarray): target values, shape (n, 1)
    global_dim, spatial_dim, local_dim (int): input feature dimensions
    global_hidden, spatial_hidden, noise_hidden (list): hidden layer sizes for each branch
    dropout_p (float): dropout rate
    mc (bool): if True, enables Monte Carlo dropout (dropout active at inference too)
    final_act (str): output activation of the mean branch
    epochs, batch_size (int): training hyperparameters

    Returns:
    model (tf.keras.Model): trained model
    history (tf.keras.callbacks.History): training history
    
    """
    model = get_model_deepsurrogate(
    global_dim=global_dim,
    spatial_dim=spatial_dim,
    local_dim=local_dim,
    global_hidden=global_hidden,
    spatial_hidden=spatial_hidden,
    noise_hidden=noise_hidden,
    dropout_p=dropout_p,
    mc=mc,
    final_act=final_act,   # 추가
)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    return model, history


def mc_predict(model, X_test, n_samples=100):
    """
    Monte Carlo Dropout based prediction with heteroscedastic noise.

    Returns:
        mean_pred (np.ndarray): predictive mean, shape (n_test,)
        epistemic_var (np.ndarray): variance from MC dropout, shape (n_test,)
        aleatoric_var (np.ndarray): averaged learned noise variance, shape (n_test,)
        total_var (np.ndarray): epistemic_var + aleatoric_var, shape (n_test,)
    """
    preds_mu, preds_sigma2 = [], []
    for _ in range(n_samples):
        out = model.predict(X_test, verbose=0)
        preds_mu.append(out[:, 0])
        preds_sigma2.append(np.exp(out[:, 1]))

    preds_mu = np.stack(preds_mu, axis=0)
    preds_sigma2 = np.stack(preds_sigma2, axis=0)

    mean_pred = preds_mu.mean(axis=0)
    epistemic_var = preds_mu.var(axis=0)
    aleatoric_var = preds_sigma2.mean(axis=0)
    total_var = epistemic_var + aleatoric_var

    return mean_pred, epistemic_var, aleatoric_var, total_var
