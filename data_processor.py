import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import librosa
import cv2
import os
from tqdm import tqdm


class PreProcessing:
    def __init__(self, data_path):
        self.bg_path = data_path + "/train/audio/_background_noise_/"
        self.bg_list = [ i for i in os.listdir(self.bg_path) if i != "README.md"]

        self.train_path = data_path + "/train/audio/"
        self.test_path = data_path + "/test/audio/"
        self.label = os.listdir(self.train_path)

        self.valid_label = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "silence"]
        self.target_label = self.valid_label + ["unknown"]
        self.lb = LabelBinarizer().fit(self.target_label)


    def _load_audio_file(self, file_path):
        input_length = 16000
        signal = librosa.core.load(file_path, sr=16000)[0]

        if len(signal) >= input_length:
            signal = signal[:input_length]
        else:
            signal = np.pad(signal, (0, 16000 - len(signal)), "constant", constant_values=0)

        return signal

    def _get_spectdata(self, signal):
        spect = librosa.feature.melspectrogram(signal, sr=16000, hop_length=161, n_fft=2048)
        log_spect = librosa.core.amplitude_to_db(spect)

        data = np.asarray(log_spect).reshape(12800)

        return data

    def _speed_tuning(self, signal):
        speed_rate = np.random.uniform(0.9, 1.1)
        wav_speed_tune = cv2.resize(signal, (1, int(len(signal) * speed_rate))).squeeze()

        if len(wav_speed_tune) < 16000:
            pad_len = 16000 - len(wav_speed_tune)
            wav_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len / 2)),
                                   wav_speed_tune,
                                   np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len / 2)))]
        else:
            cut_len = len(wav_speed_tune) - 16000
            wav_speed_tune = wav_speed_tune[int(cut_len / 2): int(cut_len / 2) + 16000]

        return wav_speed_tune

    def _pitch_tuning(self, signal):
        bins_per_octave = 24
        pitch_pm = 4
        pitch_change = pitch_pm * 2 * (np.random.uniform() - 0.5)

        wav_pitch_changed = librosa.effects.pitch_shift(signal.astype("float64"),
                                                        16000,
                                                        n_steps=pitch_change,
                                                        bins_per_octave=bins_per_octave)

        return wav_pitch_changed

    def _background_mixing(self, signal, background):
        start = np.random.randint(background.shape[0] - 16000)
        bg_slice = background[start: start + 16000]
        signal_with_bg = signal * np.random.uniform(0.8, 1.2) + bg_slice * np.random.uniform(0, 0.1)

        return signal_with_bg

    def _choice_background(self):
        return librosa.load(self.bg_path + self.bg_list[np.random.randint(0, 5)], sr=None)[0]

    def _float_feature(self, value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _make_example(selfs, feature):
        features = tf.train.Features(feature=feature)
        example = tf.train.Example(features=features)

        return example

    def processing(self, tfrecord_fname, training=True):
        if training:
            writer = tf.python_io.TFRecordWriter(tfrecord_fname)

            for label in self.label:
                if label == ".DS_Store": continue
                else:
                    orig_label = label
                    file_path = self.train_path + label + "/"
                    files = os.listdir(file_path)

                    if label not in self.valid_label:
                        label = "unknown"

                    encoded_label = self.lb.transform([label])[0]

                    progress_bar = tqdm(total=len(files), desc=f"{orig_label}")

                    for file in files:
                        if file == ".DS_Store" or file == "README.md": continue
                        else:
                            filename = file_path + file
                            signal = self._load_audio_file(filename)
                            signal_dict = {}

                            signal_dict[0] = signal
                            signal_dict[1] = self._speed_tuning(signal)
                            signal_dict[2] = self._pitch_tuning(signal)
                            signal_dict[3] = self._speed_tuning(self._pitch_tuning(signal))
                            signal_dict[4] = self._background_mixing(signal, self._choice_background())
                            signal_dict[5] = self._background_mixing(self._speed_tuning(signal), self._choice_background())
                            signal_dict[6] = self._background_mixing(self._pitch_tuning(signal), self._choice_background())
                            signal_dict[7] = self._background_mixing(self._speed_tuning(self._pitch_tuning(signal)), self._choice_background())

                            spectrums = [self._get_spectdata(signal_dict[j]) for j in signal_dict.keys()]

                            for spectrum in spectrums:
                                feature = {
                                    "spectrum": self._float_feature(spectrum),
                                    "label": self._int64_feature(encoded_label)
                                }

                                example = self._make_example(feature)
                                writer.write(example.SerializeToString())

                            progress_bar.update(1)

            writer.close()

            print("Finished")

        else:
            writer = tf.python_io.TFRecordWriter(tfrecord_fname)

            file_path = self.test_path
            files = os.listdir(file_path)

            for file in files:
                if file == ".DS_Store" or file == "README.md": continue
                else:
                    filename = file_path + file

                    signal = self._load_audio_file(filename)
                    spectrum = self._get_spectdata(signal)

                    feature = {
                        "spectrum": self._float_feature(spectrum)
                    }

                    example = self._make_example(feature)
                    writer.write(example.SerializeToString())

            writer.close()

            print("Finished")


class TFRecord:
    def __init__(self, batch_size=128):
        self.batch_size = batch_size
        self.init_op = None
        self.spectrum = None
        self.label = None

    def _train_parser(self, serialized_example):
        features = {
            "spectrum": tf.FixedLenFeature([12800], tf.float32),
            "label": tf.FixedLenFeature([12], tf.int64)
        }

        parsed_feature = tf.parse_single_example(serialized_example, features)

        spectrum = parsed_feature["spectrum"]
        label = parsed_feature["label"]

        return spectrum, label


    def _test_parser(self, serialized_example):
        features = {
            "spectrum": tf.FixedLenFeature([12800], tf.float32),
        }

        parsed_feature = tf.parse_single_example(serialized_example, features)

        spectrum = parsed_feature["spectrum"]

        return spectrum

    def make_iterator(self, tfr_fname, training=True):
        with tf.name_scope("TFRecord"):
            if training:
                data = tf.data.TFRecordDataset(tfr_fname).map(self._train_parser)
                data = data.shuffle(500000, reshuffle_each_iteration=True)
            else:
                data = tf.data.TFRecordDataset(tfr_fname).map(self._test_parser)

            data = data.batch(self.batch_size)
            iterator = tf.data.Iterator.from_structure(data.output_types, data.output_shapes)

            if training:
                spectrum, self.label = iterator.get_next()
            else:
                spectrum = iterator.get_next()

            self.spectrum = tf.cast(tf.reshape(spectrum, [-1, 128, 100, 1]), tf.float32)
            self.init_op = iterator.make_initializer(data)

    def load(self, session, training=True):
        if training:
            return session.run([self.spectrum, self.label])
        else:
            return session.run(self.spectrum)
