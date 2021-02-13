import librosa
import soundfile
import tensorflow as tf
from tf_pitch_shift import tf_pitch_shift

if __name__ == '__main__':
    sample_rate = 16000
    y, sr = librosa.load('example.wav', sr=sample_rate)
    y_tf = tf.convert_to_tensor(y)[tf.newaxis, ...]
    y_shift_librosa = librosa.effects.pitch_shift(y, sr=sample_rate, n_steps=3, res_type='fft')
    y_shift_tf = tf_pitch_shift(y_tf, sr=sample_rate, n_steps=3)
    soundfile.write('y_shift_librosa.wav', y_shift_librosa, sample_rate)
    soundfile.write('y_shift_tf.wav', y_shift_tf.numpy()[0], sample_rate)
