
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Multiply, Lambda, Concatenate, concatenate

def get_dropout(x, p=0.5, mc=False, training=None):
    return Dropout(p)(x, training=True if mc else training)


def gaussian_nll(y_true, y_pred):
    """
    Negative log-likelihood for y ~ N(mu, exp(log_sigma2)).
    y_pred: concatenated [mu, log_sigma2], shape (batch, 2)
    """
    mu = y_pred[:, 0:1]
    log_sigma2 = y_pred[:, 1:2]
    sigma2 = tf.exp(log_sigma2)
    nll = 0.5 * (log_sigma2 + tf.square(y_true - mu) / (sigma2 + 1e-8))
    return tf.reduce_mean(nll)


def mu_mse(y_true, y_pred):
    mu = y_pred[:, 0:1]
    return tf.reduce_mean(tf.square(y_true - mu))


def get_model_deepsurrogate(
    global_dim=5,
    spatial_dim=2,
    local_dim=1,
    global_hidden=[16, 8],
    spatial_hidden=[16, 8],
    noise_hidden=[32, 16],
    dropout_p=0.1,
    mc=False,
    final_act="softplus"
):
    """
    
    Args:
        global_dim (int): dimension of global feature 
        spatial_dim (int):  dimension of spatial coordinates 
        local_dim (int): dimension of local feature 
        global_hidden (list): Dense layer units for the global branch
        spatial_hidden (list): spatial term Dense layer units
        noise_hidden (list): noise branch Dense layer units
        dropout_p (float): dropout rate
        mc (bool): Monte Carlo Dropout 
        final_act (str): activation

    Returns:
        Compiled Keras model
    """

    inp_global = Input(shape=(global_dim,), name='global_inp')
    s_input = Input(shape=(spatial_dim,), name='spatial')
    local_input = Input(shape=(local_dim,), name='local')

    # Global term
    x = inp_global
    for units in global_hidden:
        x = Dense(units, activation='relu')(x)
        x = get_dropout(x, p=dropout_p, mc=mc)

    # Spatial eta term
    eta = s_input
    for units in spatial_hidden:
        eta = Dense(units, activation='relu')(eta)
        eta = get_dropout(eta, p=dropout_p, mc=mc)

    # Combine
    # Note: The final element of `global_hidden` and `spatial_hidden` must be equal, since the global and spatial branch outputs are combined via element-wise multiplication (Multiply layer).
    B_eta = Multiply(name='basis_eta_product')([x, eta])
    B_eta_sum = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='sum_B_eta')(B_eta)

    final = concatenate([B_eta_sum, local_input])
    out = Dense(1, activation=final_act)(final)

    # Log-sigma^2 term
    log_sigma2 = s_input
    for units in noise_hidden:
        log_sigma2 = Dense(units, activation='softplus')(log_sigma2)
        log_sigma2 = get_dropout(log_sigma2, p=dropout_p, mc=mc)
    log_sigma2 = Dense(1, activation='linear', name='log_sigma2')(log_sigma2)

    # noise = Lambda(lambda x: tf.exp(x), name='lognormal_noise')(log_sigma2)
    # final_output = Add(name='final_output')([out, noise])
    final_output = Concatenate(name='mu_logsigma2')([out, log_sigma2])

    model = Model(inputs=[inp_global, s_input, local_input], outputs=final_output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.96
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    model.compile(optimizer=optimizer, loss=gaussian_nll, metrics=[mu_mse])

    return model
