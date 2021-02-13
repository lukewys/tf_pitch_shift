# tf_pitch_shift
Tensorflow implementation of `librosa.effects.pitch shift`, `res_type='fft'`.

The function expect input of shape `[batch_size, T]`.

This implementation is not identical in result it generated. This is because the stft and phase vocoder implementation are different in numerical results comparing to `librosa.effects.pitch shift`.

The audio generated using this function is a bit "right shifted" comparing to librosa.

The speed of the function is not fast. In a RTX 2070, batch size=16, audio of length 65536, it takes 300ms to calculate one batch.

