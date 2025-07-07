
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, Multiply, Add, Lambda, concatenate

def get_dropout(x, p=0.5, mc=False):
    return Dropout(p)(x, training=mc)

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
        global_hidden (list): 글로벌 term Dense layer units
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
    B_eta = Multiply(name='basis_eta_product')([x, eta])
    B_eta_sum = Lambda(lambda x: tf.reduce_sum(x, axis=1, keepdims=True), name='sum_B_eta')(B_eta)

    final = concatenate([B_eta_sum, local_input])
    out = Dense(1, activation=final_act)(final)

    model = Model(inputs=[inp_global, s_input, local_input], outputs=final_output)

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.96
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])

    return model
