import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

DEFAULT_N = 128


def plot_model_3d(nn, R, T, N=DEFAULT_N, file_name='', T_l=0.0):
    """Plots the given model in 3D.
    The visualization is then for
    x from 0 to R,
    t from T_l to T,
    with N elements in between.

    Args:
        nn (tf.keras.models.Model): A Keras model, preferably the inner approximation network of a Pinn.PINN.
        R (float): The right border for x (position).
        T (float): The upper bound for t (time).
        N (int, optional): Number of values per axis in the visualization grid. Defaults to DEFAULT_N.
        file_name (str, optional): Whether to save the plot. Expects an including file extension, e.g. 'plot.pdf'. Defaults to ''.
        T_l (float, optional): The lower, left, time bound for t. Defaults to 0.0.
    """
    x_range = np.linspace(0, R, N)
    t_range = np.linspace(T_l, T, N)
    Xs, Ts = tf.cast(tf.meshgrid(x_range, t_range), tf.float32)
    Xs = tf.reshape(Xs, (-1, 1,))
    Ts = tf.reshape(Ts, (-1, 1,))
    XT = tf.concat((Xs, Ts), axis=1)

    result = nn(XT).numpy()

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x_range, t_range)
    Z = result.reshape(X.shape)

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, 40)  # Rotate a little

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$C(x, t)$')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)

    plt.title('$C(x, t)$ Neural Network')
    plt.show()


def plot_nn(nn, R, t=0.0, N=DEFAULT_N, comparison=None, file_name=''):
    """Plots the given model in 2D.
    For x from 0 to R, at time t,
    with N elements in between.

    comparison should be the final y values of the comparison.

        T (): The upper bound for t (time).
        N ([type], optional): . Defaults to DEFAULT_N.
        file_name (str, optional): Whether to save the plot. Expects an including file extension, e.g. 'plot.pdf'. Defaults to ''.
        T_l (float, optional): The lower, left, time bound for t. Defaults to 0.0.


    Args:    
        nn (tf.keras.models.Model): A Keras model, preferably the inner approximation network of a Pinn.PINN.
        R (float): The right border for x (position).
        t (float, optional): The time at which the model is printed. Defaults to 0.0.
        N (int, optional): Number of values per axis in the visualization grid. Defaults to DEFAULT_N.
        comparison (list, optional): A list of reference values. Should have N values. Defaults to None.
        file_name (str, optional): Whether to save the plot. Expects an including file extension, e.g. 'plot.pdf'. Defaults to ''.
    """
    Xs = tf.constant(np.linspace(0, R, N)[:, None], dtype=tf.float32)
    Ts = tf.constant(t, shape=Xs.shape, dtype=tf.float32)
    XT = tf.concat((Xs, Ts), axis=1)

    Z = nn(XT).numpy()[:]

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)

    ax.plot(Xs, Z, label='Prediction')
    if comparison is not None:
        ax.plot(Xs, comparison, label='Reference')

    ax.set_xlabel('x')
    ax.set_ylabel('$C(x, t)$')

    plt.legend()
    plt.title(f'$C(x, t={t:.2f})$ Neural Network')

    if file_name:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.show()


def plot_t_animation(nn, R, T, N=DEFAULT_N, T_l=0.0, y_lim=[None, None],
                     t=-1, file_name='animation.gif', num_frames=None):
    '''
    Plot an animation for
    x from 0 to R, t from 0 to T, with N elements in between.

    If the file_name is None, then only the plot at time t 
    will be shown.

    If t == -1 (unchanged) then nothing will be plotted.
    '''
    x_range = np.linspace([0], [R], N)

    Xs = tf.constant(x_range, dtype=tf.float32)
    XT = tf.concat((Xs, tf.constant(T_l, shape=Xs.shape)), axis=1)

    y = nn(XT).numpy()[:]

    # Plot the difference
    fig, ax = plt.subplots(dpi=100)
    fig.patch.set_facecolor('white')
    Cx_line, = ax.plot(x_range, y, label='C(x, t)')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$C(x, 0.0)$')
    if y_lim != [None, None]:
        ax.set_ylim(y_lim)

    if file_name is not None:
        if num_frames is None:
            num_frames = 200

        def animate(i):
            print(
                f'\rCreating Animation: {i/float(num_frames)*100:6.2f}%', end='')
            t_position = T * i/float(num_frames)
            Ts = tf.constant(t_position, dtype=tf.float32,
                             shape=Xs.numpy().shape)
            XT = tf.concat((Xs, Ts), axis=1)
            y = nn(XT).numpy()[:]
            Cx_line.set_ydata(y)
            ax.set_ylabel(f'$C(x, {t_position:5.2f})$')

        ani = matplotlib.animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=100)
        ani.save(file_name)
        print(f'\rCreating Animation: 100.00%')

    if t != -1:
        Ts = tf.constant(t, shape=Xs.numpy().shape)
        XT = tf.concat((Xs, Ts), axis=1)
        y = nn(XT)
        Cx_line.set_ydata(y)
        ax.set_ylabel('$C(x,' + str(round(t, 2)) + ')$')

        plt.legend(loc='best')
        plt.show()


def plot_difference_to_ref(nn, R, max_t, ref_f, N=DEFAULT_N, T_l=0.0,
                           title='Reference for C(x, t)', file_name=None):
    x_range = np.linspace(0, R, N)

    ts = np.linspace(T_l, max_t, 4)
    fig = plt.figure(figsize=(8, 4))
    fig.patch.set_facecolor('white')
    y_min, y_max = 99999, -999999
    axes = []
    for i in range(4):
        subplot = 221 + i
        ax = fig.add_subplot(subplot)
        axes.append(ax)

        if ref_f is not None:
            ref = ref_f(x_range, ts[i])
            y_min = np.min(ref) if np.min(ref) < y_min else y_min
            y_max = np.max(ref) if np.max(ref) > y_max else y_max
            ax.plot(x_range, ref, label='Reference')

        xt = tf.concat((x_range[:, None], tf.constant(
            ts[i], shape=x_range[:, None].shape)), axis=1)

        pred = nn(xt)
        y_min = np.min(pred) if np.min(pred) < y_min else y_min
        y_max = np.max(pred) if np.max(pred) > y_max else y_max
        ax.plot(x_range, pred, label='Prediction')
        ax.title.set_text(f't = {ts[i]:.2f}')
        ax.legend()

    for ax in axes:
        ax.set_ylim([y_min-0.1, y_max+0.1])
    plt.suptitle(title)
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight', dpi=300)
    plt.show()


def plot_ref_prediction_comparison(nn, R, T,
                                   ref_data,
                                   N=DEFAULT_N, T_l=0.0,
                                   show_error=False,
                                   title_fn_name=None, title=None,
                                   file_name=None):
    fig = plt.figure(figsize=plt.figaspect(0.33), dpi=200)
    fig.patch.set_facecolor('white')

    x_range = np.linspace(0, R, N)
    t_range = np.linspace(T_l, T, N)

    # Reference
    X, Y = np.meshgrid(x_range, t_range)
    Z = tf.reshape(ref_data, X.shape)
    Z_ref = Z

    subplot = 131 if show_error else 121
    ax = fig.add_subplot(subplot, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$C(x, t)$')
    ax.title.set_text('Reference C(x, t)')
    ax.title.set_fontweight('bold')

    # Prediction
    Xs, Ts = tf.cast(tf.meshgrid(x_range, t_range), tf.float32)
    Xs = tf.reshape(Xs, (-1, 1,))
    Ts = tf.reshape(Ts, (-1, 1,))
    XT = tf.concat((Xs, Ts), axis=1)

    Z = nn(XT).numpy()[:].reshape(X.shape)

    subplot = 133 if show_error else 122
    ax = fig.add_subplot(subplot, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, 40)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$C(x, t)$')
    ax.title.set_text('Neural Net C(x, t)')
    ax.title.set_fontweight('bold')

    # Error
    if show_error:
        Z = np.abs(Z_ref - Z)
        ax = fig.add_subplot(132, projection='3d')

        ax.plot_surface(X, Y, Z, cmap='Reds')
        ax.view_init(30, 40)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('$|C_{nn}(x, t) - C_{true}(x, t)$|')
        ax.title.set_text('Error')
        ax.title.set_fontweight('bold')

    title = 'Reference/Prediction Comparison' if title is None else title
    if title_fn_name is not None:
        title = title_fn_name + ' ' + title

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)
    plt.tight_layout()
    plt.show()


def plot_difference_noisy_true_error(nn, R, T,
                                     noisy_ref_data, true_ref_f,
                                     N=DEFAULT_N, T_l=0.0,
                                     title_fn_name=None, title=None,
                                     file_name=None):
    fig = plt.figure(dpi=200, figsize=(11, 11))
    fig.patch.set_facecolor('white')

    x_range = np.linspace(0, R, N)
    t_range = np.linspace(T_l, T, N)

    # Noisy Reference
    X, Y = np.meshgrid(x_range, t_range)
    Z = tf.reshape(noisy_ref_data, X.shape)

    ax = fig.add_subplot(221, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, 40)  # Rotate a little
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$C_{true}(x, t) + N(x, t)$')
    ax.title.set_text('Noisy Data')
    ax.title.set_fontweight('bold')

    # True Reference
    Z = tf.reshape(true_ref_f(X, Y), X.shape)
    Z_ref = Z

    ax = fig.add_subplot(223, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, 40)  # Rotate a little
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$C_{true}(x, t)$')
    ax.title.set_text('True Data')
    ax.title.set_fontweight('bold')

    # Prediction
    Xs, Ts = tf.cast(tf.meshgrid(x_range, t_range), tf.float32)
    Xs = tf.reshape(Xs, (-1, 1,))
    Ts = tf.reshape(Ts, (-1, 1,))
    XT = tf.concat((Xs, Ts), axis=1)

    Z = nn(XT).numpy()[:].reshape(X.shape)
    ax = fig.add_subplot(224, projection='3d')

    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.view_init(30, 40)  # Rotate a little
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$C_{nn}(x, t)$')
    ax.title.set_text('Neural Net')
    ax.title.set_fontweight('bold')

    # Error
    Z = np.abs(Z_ref - Z)
    ax = fig.add_subplot(222, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='Reds')
    ax.view_init(30, 40)  # Rotate a little
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('$|C_{nn}(x, t) - C_{true}(x, t)$|')
    ax.title.set_text('Error (to true data)')
    ax.title.set_fontweight('bold')

    title = 'Noisy Reference with its Prediction Comparison' if title is None else title
    if title_fn_name is not None:
        title = title_fn_name + ' ' + title

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name)

    plt.show()


# ================
# Helper Functions
# ================

def data_to_function(data, R, T, N):
    """
    Convert a training data array to a function.
    Example usage:
    plot_difference_to_ref(...,
        ref_f=data_to_function(y_train[0], R=R, T=T, N=N))

    !!! TODO: Maybe not correct !!!
    """
    def fn(x, t):
        if t < T:
            return data[int(t/T*N)*N:int(t/T*N+1)*N]
        return data[-N:]
    return fn


def function_to_data(f, R, T, N, T_l=0.0):
    """
    Convert a training function to an array.
    Example usage:
    plot_nn_reference(...,
        ref_data=function_to_data(lambda x, t: np.sin(x - 0.5*t), R=R, T=T, N=N))
    """
    xs, ts = np.meshgrid(np.linspace(0., R, N), np.linspace(T_l, T, N))
    return f(xs, ts)


# ==============
# Colorbar plots
# ==============


def plot_nn_colorbar(nn, R, T, N, T_l=0.0):
    fig = plt.figure(figsize=(8, 6))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    xs = np.linspace(0., R, N)
    ts = np.linspace(T_l, T, N)
    Xs, Ts = np.meshgrid(xs, ts)
    Z = nn(tf.concat(
        (Xs.reshape(-1, 1), Ts.reshape(-1, 1)), axis=1)
    ).numpy().reshape(Xs.shape)

    im = ax.imshow(Z, extent=[0., R, T, 0.], aspect=R/T)

    ax.set_xlabel('x')
    ax.set_ylabel('t')

    plt.colorbar(im, ticks=[np.min(Z), 0, np.max(Z)], fraction=0.046, pad=0.04)

    plt.title('C(x, t) Approximation', fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_nn_reference(nn, R, T, N, ref_data, T_l=0.0,
                      show_error=False,
                      title_fn_name=None, title=None,
                      file_name=None,
                      ratio=0.046,
                      tick_size=15):
    fig = plt.figure(figsize=plt.figaspect(0.33 if show_error else 0.5))
    fig.patch.set_facecolor('white')

    # Reference Data
    Z_ref = tf.reshape(ref_data, (N, N))

    # Prediction Data
    Xs, Ts = np.meshgrid(np.linspace(0., R, N), np.linspace(T_l, T, N))
    Z = nn(tf.concat(
        (Xs.reshape(-1, 1), Ts.reshape(-1, 1)), axis=1)
    ).numpy().reshape(Xs.shape)

    # Calculate scale
    low_tick = np.min(np.concatenate((Z, Z_ref)))
    high_tick = np.max(np.concatenate((Z, Z_ref)))
    ticks = np.linspace(low_tick, high_tick, 3)

    # Reference
    subplot = 131 if show_error else 121
    ax = fig.add_subplot(subplot)

    im_ref = ax.imshow(
        Z_ref, extent=[0., R, T, T_l], aspect=R/T, vmin=low_tick, vmax=high_tick)
    ax.set_xlabel('x', size=tick_size)
    ax.set_ylabel('t', size=tick_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    ax.set_title('Reference C(x, t)', fontsize=tick_size)
    ax.title.set_fontweight('bold')
    cb = plt.colorbar(im_ref, ticks=ticks, fraction=ratio, pad=0.04)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(tick_size)

    # Prediction
    subplot = 133 if show_error else 122
    ax = fig.add_subplot(subplot)

    im = ax.imshow(Z, extent=[0., R, T, T_l],
                   aspect=R/T, vmin=low_tick, vmax=high_tick)
    ax.set_xlabel('x', size=tick_size)
    ax.set_ylabel('t', size=tick_size)
    plt.xticks(size=tick_size)
    plt.yticks(size=tick_size)
    ax.set_title('Approximate C(x, t)', fontsize=tick_size)
    ax.title.set_fontweight('bold')
    cb = plt.colorbar(im, ticks=ticks, fraction=ratio, pad=0.04)
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(tick_size)

    # Error
    if show_error:
        Z_err = np.abs(Z_ref - Z)
        ax = fig.add_subplot(132)

        im = ax.imshow(Z_err, extent=[0., R, T, T_l], aspect=R/T, cmap='Reds')
        ax.set_xlabel('x', size=tick_size)
        ax.set_ylabel('t', size=tick_size)
        plt.xticks(size=tick_size)
        plt.yticks(size=tick_size)
        ax.set_title('$|C_{nn}(x, t) - C_{true}(x, t)$|', fontsize=tick_size)
        ax.title.set_fontweight('bold')
        cb = plt.colorbar(im, fraction=ratio, pad=0.04)
        for t in cb.ax.get_yticklabels():
            t.set_fontsize(tick_size)

    title = 'Reference/Prediction Comparison' if title is None else title
    if title_fn_name is not None:
        title = title_fn_name + ' ' + title

    plt.suptitle(title, fontweight="bold")
    plt.tight_layout()

    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    plt.show()
