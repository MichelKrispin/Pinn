import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Disable log printing

tf.get_logger().setLevel("ERROR")
tf.autograph.set_verbosity(2)

# tf.debugging.set_log_device_placement(True)
print(
    '--------------- Loaded PINN ---------------\n'
    f'GPU is{"" if tf.config.list_physical_devices("GPU") else " NOT"} '
    f'available {[elem[0] for elem in tf.config.list_physical_devices("GPU")]}'
)


def DenseNN(num_in=2, layers=[128, 128, 128], num_out=1,
            activation_fn='tanh', name='BaseNeuralNetwork'):
    """A basic, dense neural network. Usually used to approximate the solution of a PDE.

    Args:
        num_in (int, optional): The number of inputs. Defaults to 2.
        layers (list, optional): The number of hidden layers with their individual neurons. Defaults to [128, 128, 128].
        num_out (int, optional): The number of outputs. Defaults to 1.
        activation_fn (str, optional): Activation function used for all layers. Defaults to 'tanh'.
        name (str, optional): The name as used by Keras. Defaults to 'BaseNeuralNetwork'.

    Returns:
        tf.keras.models.Model: The Keras model of this network.
    """
    inputs = tf.keras.layers.Input(shape=(num_in,), name='xt_input')
    # Create the hidden layers according to the argument
    kernel_initializer = tf.keras.initializers.RandomNormal(
        mean=0.0, stddev=0.05, seed=123)
    x = inputs
    for layer in layers:
        x = tf.keras.layers.Dense(
            layer, activation=activation_fn,
            kernel_initializer=kernel_initializer
        )(x)
    outputs = tf.keras.layers.Dense(
        num_out, kernel_initializer=kernel_initializer)(x)
    return tf.keras.models.Model(
        inputs=inputs, outputs=outputs, name=name
    )


class PartialDerivativesLayer(tf.keras.layers.Layer):
    """A keras layer that only calculates the linear advection-diffusion equation.
    """

    def __init__(self, model, D, v, **kwargs):
        """Initialize the PDE layer with a model, that should approximate the PDE.
        Saves the model and the diffusion velocity coefficient.

        Args:
            model (tf.keras.models.Model): A keras network. Should have two inputs and one output.
            D (dict): A dictionary with a value and a trainable Flag: {'value': 0.1, 'trainable':True}.
                      If the trainable flag is set to true, then the value will be the initial value.
            v (dict): A dictionary with a value and a trainable Flag: {'value': 0.1, 'trainable':True}.
                      The value can either be a float or a numpy array or tensorflow tensor,
                      which is then used a a function, so it should correspond to the training data.
                      Another option is to use a function itself, e.g. lambda x, t: x + t.
                      If the trainable flag is set to true, then the value will be the initial value.

        Raises:
            TypeError: If the value of v is no float, or numpy array, or tensorflow tensor or function.
        """
        self.model = model
        self.D = tf.Variable(D['value'], trainable=D['trainable'],
                             dtype=tf.float32, name='Diffusion')

        # It is possible to have v as a function (predefined list)
        self.v_is_const = (isinstance(v['value'], float) or
                           isinstance(v['value'], np.ndarray) or
                           tf.is_tensor(v['value']))
        if self.v_is_const:
            self.v = tf.Variable(v['value'], trainable=v['trainable'],
                                 dtype=tf.float32, name='Velocity')
        elif callable(v['value']):  # Or function
            self.v = v['value']
        else:
            raise TypeError(
                f'v has to be a function or a constant (float, numpy array, tf Tensor) but is {type(v["value"])}')

        super().__init__(**kwargs)

    # There seems to be a problem with converting functions
    # that use gradient tape inside another gradient tape
    # but there is another bug that makes this crash when
    # adding the decorator below
    # @tf.autograph.experimental.do_not_convert
    def call(self, xt):
        """
        Input is x, t tensor:
        [ [x1, t1],
          [x2, t2],
          ...]

        returns: C_t - D * C_xx + v * C_x
        """
        # The outer tape has to exist for the second derivative
        with tf.GradientTape() as tape_2nd_g:
            tape_2nd_g.watch(xt)
            # The inner tape watches the first derivatives
            with tf.GradientTape() as tape_1st_g:
                tape_1st_g.watch(xt)
                C = self.model(xt)
            C_xt = tape_1st_g.batch_jacobian(C, xt)
            C_x = C_xt[..., 0]
            C_t = C_xt[..., 1]
        C_x2t2 = tape_2nd_g.batch_jacobian(C_xt, xt)
        C_xx = C_x2t2[..., 0, 0]

        # Check if v is a function or constant
        if self.v_is_const:
            return C_t - self.D * C_xx + self.v * C_x
        return C_t - self.D * C_xx + self.v(xt[..., 0], xt[..., 1]) * C_x


class PINN:
    """
    The general PINN network which holds a
    dense network and a layer that calculates the
    partial differential equations needed.

    Approximates the linear advection-diffusion equation:
    C_t - D * C_xx + v * C_x = 0.

    Can be used to learn D, v, or just generate data
    according to a given D and v.

    Note:
        Usage: `model, pinn = PINN(dense_nn,
                        D={'trainable':False, 'value':0.0},
                        v={'trainable':False, 'value':0.0}
                    ).build_model(with_data=False)`
    """

    def __init__(self, network, D, v):
        """ Initialize the pinn network.

        Args:
            network (tf.keras.models.Model): A keras network with 2 inputs, somethings in between, 1 output.
            D (dict): The diffusion coefficient, which looks like:
                {'trainable': True/False, 'value': float (initial value)}
                If trainable is False, then it is assumed that this coefficient is known.
                If it is True, then this will the initial value.
            v ([type]): The velocity coefficient, which looks exactly like D.                
        """
        super(PINN, self).__init__()

        self.net = network

        self.pd_layer = PartialDerivativesLayer(self.net, D, v)

    def build_model(self,
                    with_data=True,
                    with_initial_conditions=True,
                    with_boundaries=True,
                    periodic=True,):
        """Build a keras model according to the arguments
        and return it. A reference to this class itself shouldn't be necessary.

        Args:
            with_data (bool, optional): Whether this PINN learns with predefined data or not.
                Defaults to True.
            with_initial_conditions (bool, optional): Whether this PINN learns with initial conditions.
            with_boundaries (bool, optional): Whether this PINN learns with boundary conditions.
                Defaults to True.
            periodic (bool, optional): Whether this PINN learns with periodic boundary conditions. If the with_boundary_conditions flag is False, then this will be ignored.
                Defaults to True.

        Returns:
            tf.keras.Model: The PINN as a Keras model.
        """
        # The inputs of the network
        inputs = []

        # C(x, t) input
        in_C = None
        if with_data:
            in_C = tf.keras.layers.Input(shape=(2,), name='c_input')
            inputs.append(in_C)

        # PDE input
        in_pde = tf.keras.layers.Input(shape=(2,), name='pde_input')
        inputs.append(in_pde)

        # Initial condition C(x, 0) = f_ic(x)
        if with_initial_conditions:
            in_ic = tf.keras.layers.Input(shape=(2,), name='ic_input')
            inputs.append(in_ic)

        # Boundary condition C(x_0, t) = C(x_R, t)
        if with_boundaries:
            in_0_bc = tf.keras.layers.Input(shape=(2,), name='left_bc_input')
            inputs.append(in_0_bc)
            in_R_bc = tf.keras.layers.Input(shape=(2,), name='right_bc_input')
            inputs.append(in_R_bc)

        # The outputs of the networks
        outputs = []
        if with_data:
            # C(x, t) prediction
            outputs += [self.net(in_C)]

        # Compute gradients for the pde
        outputs += [self.pd_layer(in_pde)]

        # Initial condition guess
        if with_initial_conditions:
            outputs += [self.net(in_ic)]

        # Left boundary guess and right boundary guess
        if with_boundaries:
            outputs += [self.net(in_0_bc) - self.net(in_R_bc)]

        return tf.keras.models.Model(
            inputs=inputs, outputs=outputs, name='PINN'
        )


def get_model(D={'trainable': False, 'value': 0.0},
              v={'trainable': False, 'value': 0.5},
              with_data=True, periodic=True,
              num_in=2, layers=[128, 128, 128], num_out=1, activation_fn='tanh'):
    """Build a model with the given parameters.

    Args:
        D (dict, optional): A dictionary containing information about the diffusion coefficient. Defaults to {'trainable': False, 'value': 0.0}.
        v (dict, optional): A dictionary containing information about the velocity coefficient. Defaults to {'trainable': False, 'value': 0.5}.
        with_data (bool, optional): If a data term is used; False, if a solution to a PDE should be found. Defaults to True.
        periodic (bool, optional): Whether the boundary conditions, C(0, t) = C(R, t), are periodic. Defaults to True.
        num_in (int, optional): The number of input values, dimensions. Defaults to 2.
        layers (list, optional): The number of hidden layers with their neurons. Defaults to [128, 128, 128].
        num_out (int, optional): The number of output values, dimensions. Defaults to 1.
        activation_fn (str, optional): The activation function used for all hidden layers. Defaults to 'tanh'.

    Raises:
        TypeError: If the diffusion coefficient D is not a float.
        TypeError: If the velocity coefficient v is neither a float, numpy array, tensor, nor a function.

    Returns:
        (tf.keras.models.Model, tf.keras.models.Model): A tuple with the inner network first, and then the pinn.
            That means: `nn, pinn = get_model()`.
    """
    if not isinstance(D, dict):
        if not isinstance(D, float):
            raise TypeError(
                f'D must be either a dictionary or a float but is {type(D)}')
        D = {'trainable': False, 'value': D}
    if not isinstance(v, dict):
        if not isinstance(v, float) and not isinstance(v, np.ndarray):
            raise TypeError(
                f'v must be either a dictionary or a float but is {type(v)}')
        v = {'trainable': False, 'value': v}

    nn = DenseNN(num_in=num_in, layers=layers,
                 num_out=num_out, activation_fn=activation_fn)
    pinn = PINN(nn, D=D, v=v).build_model(
        with_data=with_data, periodic=periodic)
    return nn, pinn


def get_v(pinn):
    """Get the value of the velocity coefficient.

    Args:
        pinn (tf.keras.models.Model): The PINN as created with the `get_model` function defined in this package.

    Raises:
        KeyError: If the velocity coefficient could not be found.

    Returns:
        numpy.float32: The value of the velocity coefficient.
    """
    for var in reversed(pinn.weights):
        if 'Velocity' in var.name:
            return var.numpy()
    raise KeyError(
        f'Could not find the velocity variable in {pinn.name}. '
        'Make sure you look in the PINN for it.')


def get_D(pinn):
    """Get the value of the diffusion coefficient.

    Args:
        pinn (tf.keras.models.Model): The PINN as created with the `get_model` function defined in this package.

    Raises:
        KeyError: If the diffusion coefficient could not be found.

    Returns:
        numpy.float32: The value of the diffusion coefficient.
    """
    for var in reversed(pinn.weights):
        if 'Diffusion' in var.name:
            return var.numpy()
    raise KeyError(
        f'Could not find the diffsusion variable in {pinn.name}. '
        'Make sure you look in the PINN for it.')


# =========================
# Tests for framed learning
# =========================


def get_models(D={'trainable': False, 'value': 0.0},
               v={'trainable': False, 'value': 0.5},
               with_data=True,
               num_in=2, layers=[128, 128, 128], num_out=1,
               activation_fn='tanh'):
    """Build a model, with multiple PINNs, with the given parameters.
    Initially made to have one approximating network, that is shared by two
    different PINNs. The first PINN can be trained with an initial funciton,
    while the second PINN only takes in partial differential equation points.
    This means, that the first PINN can be first used for training and then
    the second one to train at another (maybe later) time or location.

    Args:
        D (dict, optional): A dictionary containing information about the diffusion coefficient. Defaults to {'trainable': False, 'value': 0.0}.
        v (dict, optional): A dictionary containing information about the velocity coefficient. Defaults to {'trainable': False, 'value': 0.5}.
        with_data (bool, optional): If a data term is used; False, if a solution to a PDE should be found. Defaults to True.
        num_in (int, optional): The number of input values, dimensions. Defaults to 2.
        layers (list, optional): The number of hidden layers with their neurons. Defaults to [128, 128, 128].
        num_out (int, optional): The number of output values, dimensions. Defaults to 1.
        activation_fn (str, optional): The activation function used for all hidden layers. Defaults to 'tanh'.

    Returns:
        (tf.keras.models.Model, tf.keras.models.Model): A tuple with the inner network first, and then the pinn.
            That means: `nn, pinn_ic, pinn_pde = get_model()`.
    """

    nn = DenseNN(num_in=num_in, layers=layers,
                 num_out=num_out, activation_fn=activation_fn)
    pinn_w_ic = PINN(nn, D=D, v=v).build_model(
        with_data=with_data, with_boundaries=False)
    pinn_pde = PINN(nn, D=D, v=v).build_model(
        with_data=with_data,
        with_initial_conditions=False,
        with_boundaries=False)
    return nn, pinn_w_ic, pinn_pde
