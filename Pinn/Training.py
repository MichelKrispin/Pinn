import time
import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable log printing

tf.get_logger().setLevel("ERROR")

# ==============
# Test functions
# ==============


def gaussian(x, mu, sig, R=None):
    """A gaussian bell curve.
    If R is given, then its periodic (one extension left and one right.

    Args:
        x (numpy.ndarray): The g(x) argument.
        mu (Float): The mu parameter. Position of the peak.
        sig (Float): The sig parameter. Width of the curve.
        R (Float): If given, two periodic extensions are added (one left, one right).
                   Defaults to None.

    Returns:
        numpy.ndarray: The curve for the given x positions.
    """
    if R is not None:
        left = np.exp(-np.power(x - mu + R, 2.) / (2 * np.power(sig, 2.)))
        center = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
        right = np.exp(-np.power(x - mu - R, 2.) / (2 * np.power(sig, 2.)))
        return left + center + right
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def convective_dispersion(x, t, C_0=5*1e8, D=1.0, v=0.0, x_i=0):
    """The convective-dispersion modeling equation with v = 0 from
    [Wildeboer et. al, 2018] Eq (17).

    Args:
        x (numpy.ndarray): x values.
        t (numpy.ndarray): t values.
        C_0 (, optional): Scaling coefficient. Defaults to 5*1e8.
        D (float, optional): Dispersion coefficient. Defaults to 1.0.
        v (float, optional): Velocity coefficient. Defaults to 0.0.
        x_i (int, optional): Position offset. Defaults to 0.

    Returns:
        [type]: [description]
    """
    # return (( C_0 / np.sqrt((2*np.pi*D*t)**3) ) *
    return ((C_0 / np.sqrt(4*np.pi*D*t)) *
            np.exp(
                - (x - x_i - v*t)**2 / (4*D*t)
    ))

# =============
# Training Data
# =============


def get_training_data(R, T, N, f_data, f_ic,
                      N_ic=None, N_bc=None, N_data=None,
                      data_noise=0.0, drop_values=None, T_l=0.0):
    """
    Create some training data by filling
    in the dictionary with the needed values.

    N is the number of values per row for the pde term.
    => N*N training points for the pde term.
    If the other Ns are None, then they will equal the pde N.

    f_data(x, t) is the data function evaluated
    within the borders. If None no data values will be created.
    data_noise is the data noise coefficient.
    If it is 0, no noise will be added.

    Noise Addition: (Values could be 1e-2)
    f_data + normal(0, (max(f_data) - min(f_data)) * data_noise)

    If drop_values is not None it should be a value of [0, 1).
    The percentage of how many values should be dropped.
    So 0.75 means only 25% of the data points are kept.

    f_ic is the initial condition function.
    => f_ic(x) = C(x, t=0)

    boundary_x:
        True  => C(0, t) = C(R, t)
        False => C(x, T_l) = C(x, T)
    """
    N_data = N if N_data is None else N_data
    N_ic = N if N_ic is None else N_ic
    N_bc = N if N_bc is None else N_bc

    data_dict = {}
    if f_data is not None:
        # True data training values
        data_x_range = np.linspace(0.0, R, N_data+2)[1:-1]
        data_t_range = np.linspace(T_l, T, N_data+2)[1:-1]

        if drop_values is not None:
            # If so drop some values and repeat those that still
            # exist to have the same shape as the other training values
            N_keep = int(N_data * (1.0 - drop_values))
            data_x_range = data_x_range[np.random.choice(
                N_data, N_keep, replace=False)]  # Drop some values
            data_t_range = data_t_range[np.random.choice(
                N_data, N_keep, replace=False)]
            data_x_range = np.sort(np.tile(data_x_range, int(
                np.ceil(N_data/N_keep)))[:N_data])  # Extend existing values
            data_t_range = np.sort(
                np.tile(data_t_range, int(np.ceil(N_data/N_keep)))[:N_data])

        data_x, data_t = tf.cast(tf.meshgrid(
            data_x_range, data_t_range), tf.float32)
        data_xt = tf.concat(
            (tf.reshape(data_x, (-1, 1)), tf.reshape(data_t, (-1, 1))),
            axis=1)
        if not data_noise:
            data_y = tf.constant(f_data(data_xt[..., 0], data_xt[..., 1]),
                                 dtype=tf.float32)[:, None]
        else:
            data_y_clean = f_data(data_xt[..., 0], data_xt[..., 1])
            data_y_clean += np.random.normal(0, (np.max(data_y_clean)-np.min(data_y_clean)) * data_noise,
                                             data_y_clean.shape[0])
            data_y = tf.constant(data_y_clean,
                                 dtype=tf.float32)[:, None]
        data_dict['data'] = {'xt': data_xt, 'y': data_y}

    # PDE training values without border
    pde_x_range = np.linspace(0.0, R, N+2)[1:-1]
    pde_t_range = np.linspace(T_l, T, N+2)[1:-1]
    pde_x, pde_t = tf.cast(tf.meshgrid(pde_x_range, pde_t_range), tf.float32)
    pde_xt = tf.concat(
        (tf.reshape(pde_x, (-1, 1)), tf.reshape(pde_t, (-1, 1))),
        axis=1)
    pde_y = tf.zeros(shape=(N*N, 1))
    data_dict['pde'] = {'xt': pde_xt, 'y': pde_y}

    # Initial condition C(x, 0) = C_0
    ic_x_range = np.linspace(0.0, R, N_ic)
    ic_t_range = np.full(ic_x_range.shape, T_l)
    ic_x, ic_t = tf.cast(tf.meshgrid(ic_x_range, ic_t_range), tf.float32)
    ic_xt = tf.concat(
        (tf.reshape(ic_x, (-1, 1)), tf.reshape(ic_t, (-1, 1))),
        axis=1)
    ic_C_0 = tf.constant(
        [f_ic(x) for x in tf.reshape(ic_x, [-1])],
        dtype=tf.float32)[:, None]
    data_dict['ic'] = {'xt': ic_xt, 'C_0': ic_C_0}

    # Boundary condition C(x_0, t) = C(x_R, t)
    bc_L, bc_R = None, None
    # if boundary_x:
    bc_t_range = np.linspace(T_l, T, N_bc)
    bc_x_range = np.zeros_like(bc_t_range)
    bc_x_0, bc_t = tf.cast(tf.meshgrid(bc_x_range, bc_t_range), tf.float32)
    bc_L = tf.concat(
        (tf.reshape(bc_x_0, (-1, 1)), tf.reshape(bc_t, (-1, 1))),
        axis=1)

    bc_x_R = tf.constant(R, shape=bc_x_0.shape, dtype=tf.float32)
    bc_R = tf.concat(
        (tf.reshape(bc_x_R, (-1, 1)), tf.reshape(bc_t, (-1, 1))),
        axis=1)
    """
    else:  # C(x, T_l) = C(x, T)
        bc_x_range = np.linspace(0.0, R, N_bc)
        bc_t_range = np.full(bc_x_range.shape, T_l)
        bc_x, bc_t_L = tf.cast(tf.meshgrid(bc_x_range, bc_t_range), tf.float32)
        bc_L = tf.concat(
            (tf.reshape(bc_x, (-1, 1)), tf.reshape(bc_t_L, (-1, 1))),
            axis=1)

        bc_t_T = tf.constant(T, shape=bc_t_L.shape, dtype=tf.float32)
        bc_R = tf.concat(
            (tf.reshape(bc_x, (-1, 1)), tf.reshape(bc_t_T, (-1, 1))),
            axis=1)
    """
    bc_y = tf.zeros(shape=(N_bc*N_bc, 1))
    data_dict['bc'] = {'L': bc_L, 'R': bc_R, 'y': bc_y}

    # Put everthing together
    x_train = []
    if f_data is not None:
        x_train.append(data_dict['data']['xt'])
    x_train += [
        data_dict['pde']['xt'],
        data_dict['ic']['xt'],
        data_dict['bc']['L'],
        data_dict['bc']['R']
    ]

    y_train = []
    if f_data is not None:
        y_train.append(data_dict['data']['y'])
    y_train += [
        data_dict['pde']['y'],
        data_dict['ic']['C_0'],
        data_dict['bc']['y'],
    ]

    return x_train, y_train


# ===============
# Training Helper
# ===============

class LogCallback(tf.keras.callbacks.Callback):
    """Keras callback to print nicely the current epoch with the loss value.
    """

    def __init__(self, epochs):
        super(LogCallback, self).__init__()
        self.batch = 0
        self.epoch = 0
        self.epochs = epochs
        self.first_loss = None
        self.loss = 0
        self.loading_elements = '-/|\\'

    def print_log(self):
        print('\rEpoch {:3d} {} [{}] {:.5f} [Batch {:2d}]'.format(
            self.epoch, self.loading_elements[int((self.epoch/4)) % 4],
            '=' * int(round(50*self.epoch/self.epochs)) + '>' +
            '-' * int(50 - round(50*self.epoch/self.epochs)),
            self.loss, self.batch), end='', flush=True)

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_train_end(self, logs=None):
        end_time = time.time()
        self.epoch = self.epochs
        self.batch = 0
        self.loss = logs['loss']
        self.print_log()
        print('\n')
        print(f'[{end_time - self.start_time:.2f}s] '
              f'Loss: {self.first_loss:.5f} --> {self.loss:.5f}')

    def on_train_batch_end(self, batch, logs=None):
        self.batch = batch
        if self.first_loss is None:
            self.first_loss = logs['loss']
        self.loss = logs['loss']
        self.print_log()

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if self.first_loss is None:
            self.first_loss = logs['loss']
        self.loss = logs['loss']
        self.print_log()


def mse_loss(x_train, y_train, with_data=True):
    """Custom loss to handle different sized training data.
    Not used anymore.
    """
    if with_data:
        return (
            (x_train[0] - y_train[0]) ** 2 +  # data
            (x_train[1] - y_train[1]) ** 2 +  # pde
            (x_train[2] - y_train[2]) ** 2 +  # ic
            (x_train[3] - x_train[4] - y_train[3]) ** 2  # bc
        ) / 4.
    return (
        (x_train[0] - y_train[0]) ** 2 +  # pde
        (x_train[1] - y_train[1]) ** 2 +  # ic
        (x_train[2] - x_train[3] - y_train[2]) ** 2  # bc
    ) / 3.


def train_pinn(pinn, x_train, y_train,
               optimizer='adam', loss='mse',
               batch_size=4096, epochs=100):
    """Shortcut to train the PINN using the default Keras way of training.

    Args:
        pinn (PINN): The Pinn that will be trained.
        x_train (list): List of tf.Tensor training positions.
        y_train (list): List of tf.Tensor true reference values.
        optimizer (str, optional): Passed on Keras optimizer. Defaults to 'adam'.
        loss (str, optional): Loss function. Passed to Keras compile function. Defaults to 'mse'.
        batch_size (int, optional): Batch size. Defaults to 4096.
        epochs (int, optional): Number of epochs. Defaults to 100.

    Returns:
        list: List of loss values from the training.
    """
    plateau_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1,
                                                            patience=500, cooldown=50, min_lr=1e-6, verbose=1)
    if loss == 'mse_custom':
        loss = mse_loss
    pinn.compile(optimizer=optimizer, loss=loss)
    history = pinn.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=False,
                       callbacks=[LogCallback(epochs), plateau_callback], verbose=0)
    return history.history['loss']


def plot_gradient_flow(pinn):
    pinn.trainable_weights


# =========================
# Tests for framed learning
# =========================


class TrainingData:
    """A class to wrap differently sized training data.
    Used for a training loop: `for x, y in data:`.
    """

    def __init__(self, x_train, y_train, batch_size):
        """Initialize the training data with x positions and y predictions.

        Args:
            x_train (list): List of training points. Usually [pde, ic, bc] training points.
            y_train (list): The true value for these training points.
            batch_size (Int): The batch size when running in a for loop and
                              how much training daata should be returned.
        """
        self.x_train = x_train
        self.y_train = y_train
        # Batch size is smallest length or given batch size
        # min([elem.shape[0] for elem in x_train] + [batch_size])
        self.batch_size = batch_size
        self.num_loops = int(max([elem.shape[0]
                             for elem in x_train]) / self.batch_size)

    def __iter__(self):
        for i in range(self.num_loops):
            b = self.batch_size
            yield (
                [x[(i*b) % len(x):(((i+1)*b) % len(x) if ((i+1)*b) % len(x) > 0 else len(x))]
                 for x in self.x_train],
                [y[(i*b) % len(y):(((i+1)*b) % len(y) if ((i+1)*b) % len(y) > 0 else len(y))]
                 for y in self.y_train])


@tf.function
def mse_loss_tf(pred, ref):
    """A close copy to the Keras mean squared error loss.
    Takes care of differently sized predictions/references.

    Args:
        pred (list): A list of tf.Tensor predictions. Might also be only one tf.Tensor.
        ref (list): A list of tf.Tensor true references. Might also be only one tf.Tensor.

    Returns:
        Float: A loss value.
    """
    if tf.is_tensor(pred):
        return tf.reduce_mean((pred - ref) ** 2)
    # Might be helpful to include weightung
    return sum([tf.reduce_mean((p - r)**2) for (p, r) in zip(pred, ref)])/len(pred)


@tf.function
def grad(model, inputs, targets):
    """Calculate the gradient of the models parameters with
    respect to the prediction on the inputs in comparison
    to the targets using the mean squared error.

    Args:
        model (tf.keras.Model): The model that should be optimized.
        inputs (list): A list of tf.Tensor input positions. Might also be only one tf.Tensor.
        targets (list): A list of tf.Tensor references. Might also be only one tf.Tensor.

    Returns:
        (Float, [tf.Tensor]): A tuple with the loss value and a list of gradients.  
    """
    with tf.GradientTape() as tape:
        loss_value = mse_loss_tf(model(inputs), targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def train_custom(model, train_dataset, optimizer, epochs):
    """Train the model with the custom TrainingData using MSE.

    Args:
        model (tf.keras.Model): The model that is about to learn something.
        train_dataset (TrainDataset): The TrainDataset with the training and reference data.
        optimizer (tf.keras.optimizers): The Keras optimizer class. 
        epochs (Int): Number of epochs to learn.

    Returns:
        list: List of the loss values to plot them.
    """
    loading_elements = '-/|\\'
    losses = []
    start_time = time.time()
    for epoch in range(epochs):
        l = []
        for batch, (x, y) in enumerate(train_dataset):
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            l.append(loss_value)
            print('\rEpoch {:3d} {} [{}] {:.5f} [Batch {:2d}]'.format(
                epoch, loading_elements[int((epoch/4)) % 4],
                '=' * int(round(50*epoch/epochs)) + '>' +
                '-' * int(50 - round(50*epoch/epochs)),
                loss_value, batch), end='', flush=True)
        losses.append(np.mean(l))
    print(f'\n[{time.time() - start_time:.2f}s] '
          f'Loss: {losses[0]:.5f} --> {losses[-1]:.5f}')
    return losses
