import tensorflow as tf
import numpy as np


def tf_float32(tensor):
    return tf.cast(tensor, dtype=tf.float32)


def tf_float64(tensor):
    return tf.cast(tensor, dtype=tf.float64)


def phase_vocoder(D, hop_len=None, rate=0.8):
    # Adapted from https://gist.github.com/markostam/099339af242896cd3eb335403f89603b
    # comparing to the link, add support to batch-wise input
    """Phase vocoder.  Given an STFT matrix D, speed up by a factor of `rate`.
    Based on implementation provided by:
      https://librosa.github.io/librosa/_modules/librosa/core/spectrum.html#phase_vocoder
    :param D: tf.complex64([batch_size, num_frames, num_bins]): the STFT tensor
    :param hop_len: float: the hop length param of the STFT
    :param rate: float > 0: the speed-up factor
    :return: D_stretched: tf.complex64([num_frames, num_bins]): the stretched STFT tensor
    """
    # make sure rate greater than 0
    tf.assert_greater(rate, 0.0)

    # get shape
    sh = tf.shape(D, name="STFT_shape")
    frames = sh[-2]
    fbins = sh[-1]
    nfft = tf.multiply(2, (fbins - 1), name="nfft")

    # default val
    if not hop_len:
        hop_len = tf.cast(int(nfft // 4), tf.float32)

    # make sure hop_len <= (3/4)*winlen
    tf.debugging.assert_less_equal(
        tf.cast(hop_len, tf.float32),
        tf.multiply(0.75, tf.cast(nfft, tf.float32)))

    # time steps range
    t = tf.range(0.0, tf.cast(frames, tf.float32), rate, dtype=tf.float32, name="time_steps")

    # Expected phase advance in each bin
    dphi = tf.linspace(0.0, np.pi * hop_len, fbins, name="dphi_expected_phase_advance")
    # phase_acc = tf_float32(tf.math.angle(D[0, :], name="phase_acc_init"))
    phase_acc = tf_float32(tf.math.angle(D[:, 0, :], name="phase_acc_init"))

    # Pad 0 columns to simplify boundary logic
    # D = tf.pad(D, [(0, 2), (0, 0)], mode='CONSTANT', name="padded_STFT")
    D = tf.pad(D, [(0, 0), (0, 2), (0, 0)], mode='CONSTANT', name="padded_STFT")

    # def fn(previous_output, current_input):
    def _pvoc_mag_and_cum_phase(previous_output, current_input):
        # unpack prev phase
        _, prev = previous_output

        # grab the two current columns of the STFT
        i = tf.cast((tf.floor(current_input)), tf.int64)
        # bcols = D[i:i + 2, :]
        bcols = D[:, i:i + 2, :]

        # Weighting for linear magnitude interpolation
        t_dif = current_input - tf.floor(current_input)
        # bmag = (1 - t_dif) * tf_float32(tf.math.abs(bcols[0, :])) + t_dif * tf_float32(tf.math.abs(bcols[1, :]))
        bmag = (1 - t_dif) * tf_float32(tf.math.abs(bcols[:, 0, :])) + t_dif * tf_float32(tf.math.abs(bcols[:, 1, :]))

        # Compute phase advance
        # dp = tf_float32(tf.math.angle(bcols[1, :])) - tf_float32(tf.math.angle(bcols[0, :])) - dphi
        dp = tf_float32(tf.math.angle(bcols[:, 1, :])) - tf_float32(tf.math.angle(bcols[:, 0, :])) - dphi
        dp = dp - 2 * np.pi * tf.round(dp / (2.0 * np.pi))

        # return linear mag, accumulated phase
        return bmag, prev + dp + dphi

    # initializer of zeros of correct shape for mag, and phase_acc for phase
    initializer = (tf.zeros([sh[0], fbins], tf.float32), phase_acc)
    mag, phase = tf.nest.map_structure(tf.stop_gradient, tf.scan(_pvoc_mag_and_cum_phase, t, initializer=initializer,
                                                                 parallel_iterations=10, name="pvoc_cum_phase"))

    # add the original phase_acc in
    # phase2 = tf.concat([tf.expand_dims(phase_acc, 0), phase], 0)[:-1, :]
    phase = tf.transpose(phase, (1, 0, 2))
    mag = tf.transpose(mag, (1, 0, 2))
    phase2 = tf.concat([tf.expand_dims(phase_acc, 1), phase], 1)[:, :-1, :]
    D_stretched = tf.cast(mag, tf.complex64) * tf.exp(1.j * tf.cast(phase2, tf.complex64),
                                                      name="stretched_STFT")

    return D_stretched


def tf_time_stretch(y, rate):
    """
    Tensorflow implementation of librosa.effects.time_stretch
    :param y: audio time series, in shape [batch_size,T]
    :param rate: Stretch factor. If rate > 1, then the signal is sped up. If rate < 1, then the signal is slowed down
    :return: y_stretch: audio time series stretched by the specified rate
    """
    # y: [batch_size,T]
    # rate: float
    n_fft = 2048
    hop_len = int(2048 // 4)
    # In librosa, the signal is padded so that frames are centred (centre=True)
    y_centre_pad = tf.pad(y, [[0, 0], [int(n_fft // 2), int(n_fft // 2)]], mode='REFLECT')
    D = tf.signal.stft(
        signals=y_centre_pad,
        frame_length=n_fft,
        frame_step=hop_len,
        fft_length=n_fft,
        pad_end=False)
    # The complex part of the stft is different than the librosa's. Do not know reason.
    # https://github.com/tensorflow/tensorflow/issues/16465 doesn't work.

    D_stretched = phase_vocoder(D, hop_len=None, rate=rate)

    y_stretch = tf.signal.inverse_stft(
        D_stretched,
        frame_length=2048,
        frame_step=512,
    )

    len_stretch = int(round(y.shape[-1] / rate))
    y_stretch = y_stretch[:len_stretch]

    return y_stretch


def fix_length(data, size, axis=-1):
    n = data.shape[axis]

    if n > size:
        slices = [slice(None)] * data.ndim
        slices[axis] = slice(0, size)
        return data[tuple(slices)]

    elif n < size:
        lengths = [(0, 0)] * data.ndim
        lengths[axis] = (0, size - n)
        return tf.pad(data, lengths)

    return data

def tf_resample(x, num, axis=-1):
    """
    Tensorflow implementation of fft resample in scipy.signal.resample
    Here, only real-valued x are allowed.
    The function for input "t" and "window" is not implemented.
    :param x: signal in shape [batch_size, T_stretched]
    :param num: number of samples in the resampled signal
    :param axis: axis of the signal
    :return: y: resampled signal
    """

    if 'complex' in str(x.dtype):
        raise TypeError('Complex input signal is not allowed')

    # x = np.asarray(x)
    Nx = x.shape[axis]

    # Forward transform
    X = tf.signal.rfft(x)

    # TODO: support applying window
    # # Apply window to spectrum
    # if window is not None:
    #     if callable(window):
    #         W = window(sp_fft.fftfreq(Nx))
    #     elif isinstance(window, np.ndarray):
    #         if window.shape != (Nx,):
    #             raise ValueError('window must have the same length as data')
    #         W = window
    #     else:
    #         W = sp_fft.ifftshift(get_window(window, Nx))
    #
    #     newshape_W = [1] * x.ndim
    #     newshape_W[axis] = X.shape[axis]
    #
    #     # Fold the window back on itself to mimic complex behavior
    #     W_real = W.copy()
    #     W_real[1:] += W_real[-1:0:-1]
    #     W_real[1:] *= 0.5
    #     X *= W_real[:newshape_W[axis]].reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequences (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)

    newshape[axis] = num // 2 + 1

    Y = tf.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)

    # Tensorflow wrok-around of "Y[tuple(sl)] = X[tuple(sl)]"
    Y = tf.convert_to_tensor(tf.Variable(Y)[tuple(sl)].assign(X[tuple(sl)]))

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling

            sl[axis] = slice(N // 2, N // 2 + 1)

            # Tensorflow wrok-around of "Y[tuple(sl)] *= 2."
            Y = tf.convert_to_tensor(tf.Variable(Y)[tuple(sl)].assign(Y[tuple(sl)] * 2))


        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N // 2, N // 2 + 1)

            # Tensorflow wrok-around of "Y[tuple(sl)] *= 0.5"
            Y = tf.convert_to_tensor(tf.Variable(Y)[tuple(sl)].assign(Y[tuple(sl)] * 0.5))

    # Inverse transform

    y = tf.signal.irfft(Y, tf.convert_to_tensor(num)[tf.newaxis])

    y *= (float(num) / float(Nx))

    return y

def tf_pitch_shift(y, sr, n_steps, bins_per_octave=12, axis=-1):
    """
    Implementation of librosa.effects.pitch_shift with res_type='fft'
    :param y: signal in shape [batch_size, T]
    :param sr: sample rate
    :param n_steps: how many (fractional) steps to shift ``y``
    :param bins_per_octave: how many steps per octave
    :param axis: axis of the signal
    :return: y_shift: pitch-shifted signal
    """
    if n_steps == 0:
        return y

    if bins_per_octave < 1 or not np.issubdtype(type(bins_per_octave), np.integer):
        raise ValueError("bins_per_octave must be a positive integer.")

    rate = 2.0 ** (-float(n_steps) / bins_per_octave)

    y_stretched = tf_time_stretch(y, rate)

    orig_sr = float(sr) / rate
    target_sr = sr
    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y_stretched.shape[axis] * ratio))

    y_shift = tf_resample(y_stretched, n_samples, axis=axis)

    return fix_length(y_shift, y.shape[axis], axis=axis)
