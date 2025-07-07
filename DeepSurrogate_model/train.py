import numpy as np
from .models import geta_model_deepsurrogate

def train_model_ver8(
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
    """
    model = get_model_ver8(
        global_dim=global_dim,
        spatial_dim=spatial_dim,
        local_dim=local_dim,
        global_hidden=global_hidden,
        spatial_hidden=spatial_hidden,
        noise_hidden=noise_hidden,
        dropout_p=dropout_p,
        mc=mc
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
    Monte Carlo Dropout based Prediction
    """
    predictions = []
    for _ in range(n_samples):
        pred = model.predict(X_test)
        predictions.append(pred)
    return np.array(predictions)

