import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer
import librosa
import cv2
import os
import datetime

def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def load_audio_file(file_path):
    input_length = 16000
    data = librosa.core.load(file_path, sr=16000)[0]
    if len(data) >= input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, 16000 - len(data)), "constant", constant_values=0)
    return data

def get_spectdata(wav, sr=16000, size=12800):
    spect = librosa.feature.melspectrogram(wav, sr=sr, hop_length=161, n_fft=2048)
    log_spect = librosa.core.amplitude_to_db(spect)
    data = np.asarray(log_spect).reshape(size)
    return data, log_spect

def speed_tuning(wav):
    speed_rate = np.random.uniform(0.9, 1.1)
    wav_speed_tune = cv2.resize(wav, (1, int(len(wav) * speed_rate))).squeeze()

    if len(wav_speed_tune) < 16000:
        pad_len = 16000 - len(wav_speed_tune)
        wav_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                               wav_speed_tune,
                               np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
    else:
        cut_len = len(wav_speed_tune) - 16000
        wav_speed_tune = wav_speed_tune[int(cut_len / 2): int(cut_len / 2) + 16000]
    return wav_speed_tune

def pitch_tuning(wav, sample_rate=16000):
    bins_per_octave = 24
    pitch_pm = 4
    pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)

    wav_pitch_changed = librosa.effects.pitch_shift(wav.astype('float64'),
                                                    sample_rate,
                                                    n_steps=pitch_change,
                                                    bins_per_octave=bins_per_octave)
    return wav_pitch_changed

def bg_mixing(wav, bg):
    start_ = np.random.randint(bg.shape[0] - 16000)
    bg_slice = bg[start_: start_ + 16000]
    wav_with_bg = wav * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)
    return wav_with_bg

def choice_bg():
    return librosa.load(bg_path + bg_list[np.random.randint(0, 5)], sr=None)[0]


target_label = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "unknown", "silence"]
valid_labels = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]

lb = LabelBinarizer().fit(target_label)

bg_path = './train/audio/_background_noise_/'
bg_list = os.listdir(bg_path)
bg_list.remove('README.md')

train_audio_path = './train/audio/'
test_audio_path = './test/audio/'
labels = os.listdir(train_audio_path)

train_path = "./tfrecord_new/train.tfrecord"
test_path = "./tfrecord_new/test.tfrecord"

train_writer = tf.python_io.TFRecordWriter(train_path)
test_writer = tf.python_io.TFRecordWriter(test_path)


for label in labels:
    if label == '.DS_Store':
        continue
    else:
        orig_label = label
        file_path = train_audio_path + label + "/"
        files = os.listdir(file_path)

        if label not in valid_labels:
            label = "unknown"

        encoded_label = lb.transform([label])[0]

        for file in files:
            if file == '.DS_Store' or file == 'README.md':
                continue
            else:
                filename = file_path + file

                signal = load_audio_file(filename)

                signal_dict = {}

                signal_dict[0] = signal
                signal_dict[1] = speed_tuning(signal)
                signal_dict[2] = pitch_tuning(signal)

                bg = choice_bg()
                signal_dict[3] = bg_mixing(signal, bg)

                bg = choice_bg()
                signal_dict[4] = bg_mixing((speed_tuning(signal)), bg)

                signal_dict[5] = speed_tuning(pitch_tuning(signal))

                bg = choice_bg()
                signal_dict[6] = bg_mixing(pitch_tuning(signal), bg)

                bg = choice_bg()
                signal_dict[7] = bg_mixing(speed_tuning(pitch_tuning(signal)), bg)

                specs = [get_spectdata(signal_dict[j])[0] for j in signal_dict.keys()]

                for spec in specs:
                    feature = {
                        "spectrum": float_feature(spec),
                        "label": int64_feature(encoded_label)
                    }

                    features = tf.train.Features(feature=feature)
                    example = tf.train.Example(features=features)
                    train_writer.write(example.SerializeToString())

train_writer.close()

file_path = test_audio_path
files = os.listdir(file_path)

for file in files:
    if file == '.DS_Store' or file == 'README.md':
        continue
    else:
        filename = file_path + file

        signal = load_audio_file(filename)
        spec = get_spectdata(signal)[0]

        print(spec.shape)

        feature = {
            "spectrum": float_feature(spec),
        }

        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)
        test_writer.write(example.SerializeToString())

test_writer.close()
