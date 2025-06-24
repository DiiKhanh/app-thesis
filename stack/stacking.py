# get all model 
# get all output predict
# random forest and save model in training 
# get outputs in testing
# -> meta model -> result -> voting
import os
import joblib
import torch
import torch.nn.functional as F
from torch import nn
import re
from sklearn.model_selection import train_test_split
from helper_code import *
import librosa
import mne
import numpy as np
from typing import Tuple, Dict
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from huggingface_hub import hf_hub_download


TRAIN_LOG_FILE_NAME = "training_log.csv"
VOTING_POS_MAJORITY_THRESHOLD = 0.5
DECISION_THRESHOLD = 0.5
# PARAMS_DEVICE = {"num_workers": min(26, os.cpu_count() - 2)}
PARAMS_DEVICE = {"num_workers": 0}
NO_CHANNELS_W_ARTIFACT_TO_DISCARD_EPOCH = 2  # Allowed number of channels with artifacts in an epoch to still count the epoch as good
NO_CHANNELS_W_ARTIFACT_TO_DISCARD_WINDOW = 4  # Allowed number of channels with artifacts in a window to still count the window as good and replace the channels with artifacts by a random other channel
RESAMPLING_FREQUENCY = 128
FILTER_SIGNALS = True
SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING = 120
EPOCH_SIZE_FILTER = (
    10  # seconds   # Epoch size to use for artifact detection within a window
)
STRIDE_SIZE_FILTER = 1
WINDOW_SIZE_FILTER = 5
LOW_THRESHOLD = -300
HIGH_THRESHOLD = 300
BIPOLAR_MONTAGES = (
    None  # Not used in torch. Must have the format [[ch1, ch2], [ch3, ch4], ...]
)
AGG_OVER_CHANNELS = True
AGG_OVER_TIME = True

EEG_CHANNELS = [
    "Fp1",
    "Fp2",
    "F7",
    "F8",
    "F3",
    "F4",
    "T3",
    "T4",
    "C3",
    "C4",
    "T5",
    "T6",
    "P3",
    "P4",
    "O1",
    "O2",
    "Fz",
    "Cz",
    "Pz",
] 

USE_TORCH = True
USE_GPU = True
PRETRAIN_MODEL_FILEPATH = "../../pretrain_models/convnext_tiny-983f1562.pth"
PARAMS_TORCH = {
    "batch_size": 16,
    "val_size": 0.2,
    "max_epochs": 10,
    "pretrained": True,
    "learning_rate": 0.00005,
}
LIM_HOURS_DURING_TRAINING = True
NUM_HOURS_EEG_TRAINING = -72
INFUSE_STATIC_FEATURES = False
PREPROCESSED_DATA_FOLDER = "../../../../preprocessed_data"
NUM_HOURS_EEG = -72

class RecordingsDataset(Dataset):
    def __init__(
        self,
        data_folder,
        patient_ids,
        device,
        group="EEG",
        load_labels: bool = True,
        hours_to_use: int = None,
        raw: bool = False,
    ):

        self.raw = raw
        self._precision = torch.float32
        self.hours_to_use = hours_to_use
        self.group = group
        if self.group == "EEG":
            self.channels_to_use = EEG_CHANNELS
        else:
            raise NotImplementedError(f"Group {self.group} not implemented.")

        # Load labels and features
        recording_locations_list = list()
        patient_ids_list = list()
        labels_list = list()
        features_list = list()
        for patient_id in patient_ids:
            patient_metadata = load_challenge_data(data_folder, patient_id)
                
            if INFUSE_STATIC_FEATURES:
                (
                    current_features,
                    current_feature_names,
                    hospital,
                    recording_infos,
                ) = get_features(data_folder, patient_id, recording_features = False, normalize = True)
            else:
                current_features = np.nan
            recording_ids = find_recording_files(
                data_folder, patient_id
            )
            
            
            if self.hours_to_use is not None:
                if abs(self.hours_to_use) < len(recording_ids):
                    if self.hours_to_use > 0:
                        recording_ids = recording_ids[: self.hours_to_use]
                    else:
                        recording_ids = recording_ids[self.hours_to_use :]
                        
            if load_labels:
                current_outcome = get_outcome(patient_metadata)
            else:
                current_outcome = 0
                
                
            for recording_id in recording_ids:
                if not is_nan(recording_id):
                    recording_location_aux = os.path.join(
                        data_folder,
                        patient_id,
                        "{}_{}".format(recording_id, self.group),
                    )
                    if os.path.exists(recording_location_aux + ".hea"):
                        recording_locations_list.append(recording_location_aux)
                        patient_ids_list.append(patient_id)
                        labels_list.append(current_outcome)
                        features_list.append(current_features)

                        if INFUSE_STATIC_FEATURES:
                            self.num_additional_features = len(current_features)
                        else:
                            self.num_additional_features = 0
                        
        
        self.recording_locations = recording_locations_list
        self.patient_ids = patient_ids_list
        self.labels = labels_list
        self.features = features_list
        self.device = device

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load the data if exist
        file_path = self.recording_locations[idx].replace("../../../../training/", "")
        preprocessed_signal_data_filepath = os.path.join(PREPROCESSED_DATA_FOLDER, file_path+".npy")
        signal_data = None
        sampling_frequency = RESAMPLING_FREQUENCY
        
        if os.path.exists(preprocessed_signal_data_filepath):
            signal_data = read_signal_data_from_file(preprocessed_signal_data_filepath)
        else: 
        # save signal_data if it not exist
            try:
                (
                    signal_data, #(19,38000) - fs = 60 - ra được đoạn 5 phút ok
                    signal_channels,
                    sampling_frequency,
                ) = load_recording_data_wrapper(
                    self.recording_locations[idx], self.channels_to_use
                )
            
                
            except Exception as e:
                print("Error loading {}".format(self.recording_locations[idx]))
                return None
            
            hea_file = load_text_file(self.recording_locations[idx] + ".hea")
            utility_frequency = get_utility_frequency(hea_file)
            signal_data, signal_channels = reduce_channels(
                signal_data, signal_channels, self.channels_to_use
            )

            # Preprocess the data.
            signal_data, sampling_frequency = preprocess_data(
                signal_data, sampling_frequency, utility_frequency
            )
            
            save_signal_data_of_one_record(preprocessed_signal_data_filepath, signal_data)
        

        # Get the other information.
        id = self.patient_ids[idx]
        hea_file = load_text_file(self.recording_locations[idx] + ".hea")
        hour = get_hour(hea_file)
        quality = get_quality(hea_file)

        # Get the label
        label = self.labels[idx]
        label = torch.from_numpy(np.array(label).astype(np.float32)).to(self._precision)

        # Get the static features.
        static_features = self.features[idx]
        static_features = np.nan_to_num(static_features, nan=-1)
        static_features = torch.from_numpy(np.array(static_features).astype(np.float32)).to(self._precision)

        # just for machine learning
        if self.raw:
            return_dict = {
                "signal": signal_data,
                "features": static_features,
                "label": self.labels[idx],
                "id": id,
                "hour": hour,
                "quality": quality,
            }

            return return_dict

        target_size = 901
        hop_length = max(int(round(signal_data.shape[1] / target_size, 0)), 1)

        # Get the spectrograms.
        n_fft = 2**10
        if signal_data.shape[1] < n_fft:
            pad_length = n_fft - signal_data.shape[1]
            padded_signal_data = np.pad(signal_data, ((0, 0), (0, pad_length)))
            print(
                f"Padding signal {self.recording_locations[idx]} of length {signal_data.shape[1]} with {pad_length} zeros for n_fft {n_fft}"
            )
        else:
            padded_signal_data = signal_data
            
        spectrograms = librosa.feature.melspectrogram(
            y=padded_signal_data,
            sr=sampling_frequency,
            n_mels=224,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        spectrograms = torch.from_numpy(spectrograms.astype(np.float32))
        spectrograms = nn.functional.normalize(spectrograms).to(self._precision)
        spectrograms = spectrograms.unsqueeze(0)
        spectrograms_resized = F.interpolate(
            spectrograms,
            size=(spectrograms.shape[2], target_size),
            mode="bilinear",
            align_corners=False,
        )
        spectrograms = spectrograms_resized.squeeze(0)

        # we have to return this
        return_dict = {
            "image": spectrograms.to(self._precision),
            "features": static_features,
            "label": label.to(self._precision),
            "id": id,
            "hour": hour,
            "quality": quality,
        }

        return return_dict
def get_quality(string):
    start_time_sec = convert_hours_minutes_seconds_to_seconds(*get_start_time(string))
    end_time_sec = convert_hours_minutes_seconds_to_seconds(*get_end_time(string))
    quality = (end_time_sec - start_time_sec) / 3600
    return quality


def get_hour(string):
    hour_start = get_start_time(string)[0]
    return hour_start
def preprocess_data(data, sampling_frequency, utility_frequency):
    # Define the bandpass frequencies.
    passband = [0.5, 30.0]

    # Promote the data to double precision because these libraries expect double precision.
    data = np.asarray(data, dtype=np.float64)

    # If the utility frequency is between bandpass frequencies, then apply a notch filter.
    if (
        utility_frequency is not None
        and passband[0] <= utility_frequency <= passband[1]
    ):
        data = mne.filter.notch_filter(
            data, sampling_frequency, utility_frequency, n_jobs=1, verbose="error"
        )

    # Apply a bandpass filter.
    data = mne.filter.filter_data(
        data, sampling_frequency, passband[0], passband[1], n_jobs=1, verbose="error"
    )

    # Resample the data.
    resampling_frequency = RESAMPLING_FREQUENCY
    lcm = np.lcm(int(round(sampling_frequency)), int(round(resampling_frequency)))
    up = int(round(lcm / sampling_frequency))
    down = int(round(lcm / resampling_frequency))
    data = scipy.signal.resample_poly(data, up, down, axis=1)

    # Scale the data to the interval [-1, 1].
    min_value = np.min(data)
    max_value = np.max(data)
    if min_value != max_value:
        data = 2.0 / (max_value - min_value) * (data - 0.5 * (min_value + max_value))
    else:
        data = 0 * data

    return data, resampling_frequency

def compute_score(
    signal: np.ndarray,
    signal_frequency: float,
    epoch_size: int,
    window_size: int,
    stride_length: int,
    high_threshold: float = None,
    low_threshold: float = None,
) -> Tuple[Dict, Dict]:
    """
    Computes a score for each window in the EEG signal.

    Parameters
    ----------
    signal: np.ndarray
        The EEG signal as a 2D numpy array of shape (channel, num_observations).
    signal_frequency: float
        The frequency of the signal in Hz.
    epoch_size: int
        The size of each epoch in seconds.
    window_size: int
        The size of each window in minutes.
    stride_length: int
        The stride length in minutes for moving the window.
    high_threshold: float
        The upper limit for detecting extreme values.
    low_threshold: float
        The lower limit for detecting extreme values.

    Returns
    -------
    Tuple[Dict, Dict]
        A tuple of two dictionaries. The first dictionary contains the score for each window. The second dictionary contains the channels to be replaced for each window.
    """
    num_channels, num_observations = signal.shape
    epoch_samples = int(signal_frequency * epoch_size)
    window_samples = int(signal_frequency * window_size * 60)
    stride_samples = int(signal_frequency * stride_length * 60)
    scores = {}
    channels_to_replace = {}
    if window_samples > num_observations:
        window_samples = num_observations
    if stride_samples > num_observations:
        stride_samples = num_observations
    for start in range(0, num_observations - window_samples + 1, stride_samples):
        window = signal[:, start : start + window_samples]
        if epoch_samples > window_samples:
            epoch_samples = window_samples
        artifact_epochs = 0
        total_epochs = 0
        ignored_artifcat_epochs = 0
        channels_with_artifacts = []
        for epoch_start in range(0, window_samples - epoch_samples + 1, epoch_samples):
            no_channels_with_artifacts = 0
            artifact_epochs_old = artifact_epochs
            for channel in range(num_channels):
                epoch = window[channel, epoch_start : epoch_start + epoch_samples]
                if check_artifacts(epoch):
                    no_channels_with_artifacts += 1
                    channels_with_artifacts.append(channel)
                if no_channels_with_artifacts > NO_CHANNELS_W_ARTIFACT_TO_DISCARD_EPOCH:
                    artifact_epochs += 1
                    break  # if any channel in the epoch has artifact, consider the whole epoch contaminated
            if artifact_epochs == artifact_epochs_old:
                if no_channels_with_artifacts > 0:
                    ignored_artifcat_epochs += 1
            total_epochs += 1
        if len(set(channels_with_artifacts)) > NO_CHANNELS_W_ARTIFACT_TO_DISCARD_WINDOW:
            artifact_epochs = artifact_epochs + ignored_artifcat_epochs
            channels_with_artifacts = []
        score = 1 - artifact_epochs / total_epochs
        # Converting start sample index to time in seconds
        start_time = start / signal_frequency
        scores[start_time] = score
        channels_to_replace[start_time] = list(set(channels_with_artifacts))
    return scores, channels_to_replace


def check_artifacts(
    epoch: np.ndarray,
    low_threshold: float = LOW_THRESHOLD,
    high_threshold: float = HIGH_THRESHOLD,
) -> bool:
    """
    Checks an EEG epoch for artifacts.

    Parameters
    ----------
    epoch: np.ndarray
        The EEG signal epoch as a 1D numpy array.
    low_threshold: float
        The lower limit for detecting extreme values.
    high_threshold: float
        The upper limit for detecting extreme values.

    Returns
    -------
    bool
        True if any type of artifact is detected, False otherwise.
    """
    # Flat signal
    if np.all(epoch == epoch[0]):
        return True

    # Extreme high or low values
    if np.max(epoch) > high_threshold or np.min(epoch) < low_threshold:
        return True

    return False


def load_recording_data_wrapper(record_name, channels_to_use):
    """
    Loads the EEG signal of a recording and applies artifact removal.
    """

    window_size = WINDOW_SIZE_FILTER  # minutes
    stride_length = STRIDE_SIZE_FILTER  # minutes
    epoch_size = EPOCH_SIZE_FILTER  # seconds

    signal_data, signal_channels, sampling_frequency = load_recording_data(record_name)

    # Remove the first and last x seconds of the recording to avoid edge effects.
    num_samples_to_remove = int(
        SECONDS_TO_IGNORE_AT_START_AND_END_OF_RECORDING * sampling_frequency
    )
    if num_samples_to_remove > 0:
        if signal_data.shape[1] > (
            2 * num_samples_to_remove + sampling_frequency * 60 * 5
        ):
            signal_data = signal_data[:, num_samples_to_remove:-num_samples_to_remove]

    if FILTER_SIGNALS:
        scores, channels_to_replace = compute_score(
            signal_data,
            signal_frequency=sampling_frequency,
            epoch_size=epoch_size,
            window_size=window_size,
            stride_length=stride_length,
        )
        signal_data_filtered = keep_best_window(
            signal=signal_data,
            scores=scores,
            signal_frequency=sampling_frequency,
            window_size=window_size,
            channels_to_replace=channels_to_replace,
            channels_to_use=channels_to_use,
        )
        signal_return = signal_data_filtered

    else:
       
        signal_return = signal_data
    return signal_return, signal_channels, sampling_frequency

def keep_best_window(
    signal: np.ndarray,
    scores: Dict[int, float],
    signal_frequency: float, #128
    window_size: int, #5
    channels_to_replace: Dict[int, float],
    channels_to_use: list()
) -> np.ndarray:
    """
    Keep only the window with the best score in the EEG signal and set all other samples to NaN.

    Parameters
    ----------
    signal: np.ndarray
        The EEG signal as a 2D numpy array of shape (channel, num_observations).
    scores: Dict[int, float]
        A dictionary where keys are the starting time (in seconds) of each window and values are the corresponding scores.
    signal_frequency: float
        The frequency of the signal in Hz.
    window_size: int
        The size of each window in minutes.
    channels_to_replace: Dict[int, float]
        A dictionary where keys are the starting time (in seconds) of each window and values are the corresponding channels to be replace.
    channels_to_use: list()
        A list of channels to use.

    Returns
    -------
    np.ndarray
        The EEG signal where only the window with the best score is kept and all other samples are set to NaN.
    """
    num_channels, num_observations = signal.shape
    window_samples = int(signal_frequency * window_size * 60) # ~ 5 min

    # Find the window with the best score
    best_start_time = get_max_key(scores)
    best_start_sample = int(best_start_time * signal_frequency)

    # Check if channels_to_replace contains for best_start_time channels that have to be replaced
    new_signal = signal.copy()
    if len(channels_to_replace[best_start_time]) > 0:
        # Replace the channels that have to be replaced with any random channel from channels_to_use that is not in channels_to_replace[best_start_time]
        for channel in channels_to_replace[best_start_time]:
            random_channel = random.choice(
                [
                    c
                    for c in range(signal.shape[0])
                    if c not in channels_to_replace[best_start_time]
                ]
            )
            new_signal[channel, :] = signal[random_channel, :]

    if best_start_sample + window_samples > num_observations:
        new_signal = new_signal[:, best_start_sample:]
    else:
        new_signal = new_signal[
            :, best_start_sample : best_start_sample + window_samples
        ]

    return new_signal

def get_max_key(score: Dict):
    """
    Returns the key with the maximum value in a dictionary.
    """

    # Get the maximum value
    max_value = max(score.values())

    # Get a list of keys with max value
    max_keys = [key for key, value in score.items() if value == max_value]

    # Calculate the middle index
    mid = len(max_keys) // 2

    # If the list has an even number of elements
    if len(max_keys) % 2 == 0:
        # Return a random key from the two middle ones
        return random.choice(max_keys[mid - 1 : mid + 1])
    else:
        # Otherwise, return the key at the middle index
        return max_keys[mid]


def get_correct_hours(num_hours):
    if num_hours < 0:
        use_last_hours_recording = True
        hours_recording = -num_hours
        start_recording = 1
    elif num_hours > 0:
        use_last_hours_recording = False
        hours_recording = num_hours
        start_recording = 0
    else:
        raise ValueError("num_hours should be either positive or negative.")
    return use_last_hours_recording, hours_recording, start_recording


def get_features(data_folder, patient_id, return_as_dict=False, recording_features=True, normalize = False):
    # Load patient data.
    patient_metadata = load_challenge_data(data_folder, patient_id)
    recording_ids_eeg = find_recording_files(data_folder, patient_id, "EEG")
    use_last_hours_eeg, hours_eeg, start_eeg = get_correct_hours(NUM_HOURS_EEG)

    # Extract patient features.
    patient_features, patient_feature_names = get_patient_features(
        patient_metadata,
        recording_ids_eeg,
        normalize = normalize
    )
    hospital = get_hospital(patient_metadata)

    # Extract recording features.
    feature_values = patient_features
    feature_names = patient_feature_names
    recording_infos = {}
    if recording_features:
        feature_types = ["EEG"]
        use_flags = {"EEG": True}
        starts = {
            "EEG": start_eeg,
        }
        hours = {"EEG": hours_eeg}
        use_last_hours = {
            "EEG": use_last_hours_eeg,
        }
        recording_ids = {
            "EEG": recording_ids_eeg,
        }
        channels_to_use = {
            "EEG": EEG_CHANNELS,
        }
        for feature_type in feature_types:
            if use_flags[feature_type]:
                feature_data = process_recording_feature(
                    feature_type,
                    starts[feature_type],
                    hours[feature_type],
                    use_last_hours[feature_type],
                    recording_ids[feature_type],
                    channels_to_use[feature_type],
                    data_folder,
                    patient_id,
                )
                recording_infos.update(feature_data)
                feature_values = np.hstack(
                    (feature_values, np.hstack(feature_data[f"{feature_type}_features"]))
                )
                feature_names = np.hstack(
                    (
                        feature_names,
                        np.hstack(feature_data[f"{feature_type}_feature_names"]),
                    )
                )

    if return_as_dict:
        return {k: v for k, v in zip(feature_names, feature_values)}
    else:
        return feature_values, feature_names, hospital, recording_infos
    
def get_eeg_features(data, sampling_frequency):
    num_channels, num_samples = np.shape(data)

    if num_samples > 0:
        delta_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=0.5, fmax=4.0, verbose=False
        )
        theta_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=4.0, fmax=8.0, verbose=False
        )
        alpha_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=8.0, fmax=12.0, verbose=False
        )
        beta_psd, _ = mne.time_frequency.psd_array_welch(
            data, sfreq=sampling_frequency, fmin=12.0, fmax=30.0, verbose=False
        )

        delta_psd_mean = np.nanmean(delta_psd, axis=1)
        theta_psd_mean = np.nanmean(theta_psd, axis=1)
        alpha_psd_mean = np.nanmean(alpha_psd, axis=1)
        beta_psd_mean = np.nanmean(beta_psd, axis=1)
    else:
        delta_psd_mean = theta_psd_mean = alpha_psd_mean = beta_psd_mean = float(
            "nan"
        ) * np.ones(num_channels)
    features = np.array(
        (delta_psd_mean, theta_psd_mean, alpha_psd_mean, beta_psd_mean)
    ).T
    features = features.flatten()

    return features

def get_recording_features(
    recording_ids, recording_id_to_use, data_folder, patient_id, group, channels_to_use
):
    if group == "EEG":
        if BIPOLAR_MONTAGES is not None:
            channel_length = len(BIPOLAR_MONTAGES)
        else:
            channel_length = len(channels_to_use)
        dummy_channels = channel_length * 4
        recording_feature_names = np.array(
            (
                np.array(
                    [
                        f"delta_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"theta_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"alpha_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"beta_psd_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
            )
        ).T.flatten()
    elif (group == "ECG") or (group == "REF") or (group == "OTHER"):
        channel_length = len(channels_to_use)
        dummy_channels = channel_length * 2
        recording_feature_names = np.array(
            (
                np.array(
                    [
                        f"{group}_mean_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
                np.array(
                    [
                        f"{group}_std_c_{i}_hour_{recording_id_to_use}"
                        for i in range(channel_length)
                    ]
                ),
            )
        ).T.flatten()
    else:
        raise ValueError("Group should be either EEG, ECG, REF or OTHER")

    if len(recording_ids) > 0:
        if abs(recording_id_to_use) <= len(recording_ids):
            recording_id = recording_ids[recording_id_to_use]
            recording_location = os.path.join(
                data_folder, patient_id, "{}_{}".format(recording_id, group)
            )
            if os.path.exists(recording_location + ".hea"):
                data, channels, sampling_frequency = load_recording_data_wrapper(
                    recording_location, channels_to_use
                )
                hea_file = load_text_file(recording_location + ".hea")
                utility_frequency = get_utility_frequency(hea_file)
                quality = get_quality(hea_file)
                hour = get_hour(hea_file)
                if (
                    all(channel in channels for channel in channels_to_use)
                    or group != "EEG"
                ):
                    data, channels = reduce_channels(data, channels, channels_to_use)
                    data, sampling_frequency = preprocess_data(
                        data, sampling_frequency, utility_frequency
                    )
                    if group == "EEG":
                        if BIPOLAR_MONTAGES is not None:
                            data = np.array(
                                [
                                    data[channels.index(montage[0]), :]
                                    - data[channels.index(montage[1]), :]
                                    for montage in BIPOLAR_MONTAGES
                                ]
                            )
                        recording_features = get_eeg_features(data, sampling_frequency)
                    
                    else:
                        raise NotImplementedError(f"Group {group} not implemented.")
                else:
                    print(
                        f"For patient {patient_id} recording {recording_id} the channels {channels_to_use} are not all available. Only {channels} are available."
                    )
                    recording_features = float("nan") * np.ones(
                        dummy_channels
                    )  # 2 bipolar channels * 4 features / channel
            else:
                recording_features = float("nan") * np.ones(
                    dummy_channels
                )  # 2 bipolar channels * 4 features / channel
                sampling_frequency = (
                    utility_frequency
                ) = channels = quality = hour = np.nan
        else:
            recording_features = float("nan") * np.ones(
                dummy_channels
            )  # 2 bipolar channels * 4 features / channel
            sampling_frequency = (
                utility_frequency
            ) = channels = recording_id = quality = hour = np.nan
    else:
        recording_features = float("nan") * np.ones(
            dummy_channels
        )  # 2 bipolar channels * 4 features / channel
        sampling_frequency = (
            utility_frequency
        ) = channels = recording_id = quality = hour = np.nan

    # Aggregate over channels
    if AGG_OVER_CHANNELS:
        recording_feature_group_names = [f'{f.split("_c_")[0]}_hour_{f.split("_c_")[-1].split("hour_")[-1]}' for f in recording_feature_names]
        recording_features, recording_feature_names = aggregate_features(recording_features, recording_feature_names, recording_feature_group_names)
        channels = [f"Agg_over_{int(len(recording_feature_group_names)/len(recording_feature_names))}_channels"]

    return (
        recording_features,
        recording_feature_names,
        sampling_frequency,
        utility_frequency,
        channels,
        recording_id,
        quality,
        hour,
    )


def aggregate_features(recording_features, recording_feature_names, recording_feature_group_names):
    unique_groups = np.unique(recording_feature_group_names)
    recording_features_agg = np.zeros(len(unique_groups))
    for i, group in enumerate(unique_groups):
        aux_values = recording_features[[True if group == f else False for f in recording_feature_group_names]]
        if len(aux_values) == 0 or np.all(np.isnan(aux_values)):
            recording_features_agg[i] = float("nan")
        else:
            recording_features_agg[i] = np.nanmean(aux_values)
    recording_features = recording_features_agg
    recording_feature_names = unique_groups

    return recording_features, recording_feature_names

def process_recording_feature(
    feature_type,
    start,
    hours,
    use_last_hours,
    recording_ids,
    channels_to_use,
    data_folder,
    patient_id,
):
    feature_data = {
        f"{feature_type}_{item}": []
        for item in [
            "features",
            "feature_names",
            "sampling_frequency",
            "utility_frequency",
            "channels",
            "recording_id",
            "quality",
            "hour",
        ]
    }

    for h in range(start, hours + start):
        if use_last_hours:
            h_to_use = -h
        (
            features,
            feature_names,
            sampling_frequency,
            utility_frequency,
            channels,
            recording_id,
            quality,
            hour,
        ) = get_recording_features(
            recording_ids=recording_ids,
            recording_id_to_use=h_to_use,
            data_folder=data_folder,
            patient_id=patient_id,
            group=feature_type.upper(),
            channels_to_use=channels_to_use,
        )

        for item in feature_data.keys():
            feature_data[item].append(locals()[item.split("_", 1)[-1]])

    if AGG_OVER_TIME:
        feature_names_aux = np.array([item for row in feature_data[f"{feature_type}_feature_names"] for item in row])
        features_aux = np.array([item for row in feature_data[f"{feature_type}_features"] for item in row])
        recording_feature_group_names = [f'{f.split("_hour_")[0]}' for f in feature_names_aux]
        feature_data[f"{feature_type}_features"], feature_data[f"{feature_type}_feature_names"] = aggregate_features(features_aux, feature_names_aux, recording_feature_group_names)
        feature_data[f"{feature_type}_sampling_frequency"] = feature_data[f"{feature_type}_sampling_frequency"][0]
        feature_data[f"{feature_type}_utility_frequency"] = feature_data[f"{feature_type}_utility_frequency"][0]
        feature_data[f"{feature_type}_channels"] = feature_data[f"{feature_type}_channels"][0]
        value_aux = feature_data[f"{feature_type}_recording_id"][0]
        if isinstance(value_aux, str):
            feature_data[f"{feature_type}_recording_id"] = value_aux.split("_")[0]
        elif value_aux is np.nan:
            feature_data[f"{feature_type}_recording_id"] = value_aux
        else:
            raise ValueError(f"Unexpected value for recording_id: {value_aux}")
        feature_data[f"{feature_type}_hour"] = f'agg_from_{np.min(feature_data[f"{feature_type}_hour"])}_to_{np.max(feature_data[f"{feature_type}_hour"])}'
        feature_data[f"{feature_type}_quality"] = np.nanmean(feature_data[f"{feature_type}_quality"])

    return feature_data

def get_patient_features(
    data, recording_ids_eeg, normalize = False
):
    age = get_age(data)
    sex = get_sex(data)
    rosc = get_rosc(data)
    ohca = get_ohca(data)
    shockable_rhythm = get_shockable_rhythm(data)
    ttm = get_ttm(data)

    sex_features = np.zeros(2, dtype=int)
    if sex == "Female":
        female = 1
        male = 0
        other = 0
    elif sex == "Male":
        female = 0
        male = 1
        other = 0
    else:
        female = 0
        male = 0
        other = 1

    if len(recording_ids_eeg) > 0:
        last_eeg_hour = np.max([int(r.split("_")[-1]) for r in recording_ids_eeg])
    else:
        last_eeg_hour = np.nan

    # Get binary features wheather the different signals are available.
    eeg_available = 1 if len(recording_ids_eeg) > 0 else 0

    # Normalize
    if normalize:
        age = age / 100
        rosc = rosc / 200
        ttm = ttm / 36
        last_eeg_hour = last_eeg_hour / 72

    features = np.array(
        (
            age,
            female,
            male,
            #other,
            rosc,
            ohca,
            shockable_rhythm,
            ttm,
            last_eeg_hour,
        )
    )
    feature_names = [
        "age",
        "female",
        "male",
        #"other",
        "rosc",
        "ohca",
        "shockable_rhythm",
        "ttm",
        "last_eeg_hour"
    ]

    return features, feature_names


class TorchvisionModel(torch.nn.Module):
    def __init__(
        self,
        model_name,
        num_classes=2,
        classification="binary",
        print_freq=100,
        batch_size=10,
        d_size=500,
        pretrained=False,
        channel_size=3,
    ):
        super().__init__()
        self._d_size = d_size
        self._b_size = batch_size
        self._print_freq = print_freq
        self.model_name = model_name
        self.num_classes = num_classes
        self.classification = classification
        self.model = eval(f"models.{model_name}()")
        
        if pretrained:
            print(f"Using pretrained {model_name} model")
            state_dict = torch.load(PRETRAIN_MODEL_FILEPATH)
            state_dict = self.update_densenet_keys(state_dict)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Using {model_name} model from scratch")
            
        

        # add a linear layer for additional features
        # A[Input Image] --> B[CNN Layers]
        # C[Additional Features] --> D[Additional Layer]
        # B --> E[Image Features]
        # D --> F[Processed Additional Features]
        # E --> G[Concatenate]
        # F --> G
        # G --> H[Final Classification Layer]
        print("model_name: ", self.model_name)

        if "resnet" in model_name.lower():
            self.model.conv1 = nn.Conv2d(channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features , self.num_classes)
        elif "convnext" in model_name.lower():
            self.model.features[0][0] = torch.nn.Conv2d(19, 96, kernel_size=(4, 4), stride=(4, 4))
            num_features = self.model.classifier[2].in_features
            self.model.classifier = nn.Sequential(
       nn.LayerNorm(num_features, eps=1e-6),
       nn.Linear(num_features, self.num_classes)
   )
        elif "efficientnet" in model_name.lower():
            self.model.features[0]= nn.Conv2d(channel_size, 24, kernel_size=7, stride=2, padding=3, bias=False)
            num_features = self.model.classifier[1].in_features
            self.model.classifier = nn.Linear(num_features, self.num_classes)
        elif "densenet" in model_name.lower():
            self.model.features[0] = nn.Conv2d(
            channel_size, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
            num_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(num_features, self.num_classes)
        
        
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")

    def update_densenet_keys(self, state_dict):
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        return state_dict
    
    def forward(self, x, classify=True):
        if "convnext" in self.model_name.lower():
            x = self.model.features(x)  
            x = self.model.avgpool(x)   
            x = x.view(x.size(0), -1)   
            x = self.model.classifier(x)
        else:
            x = self.model.forward(x)
        return x

    def unpack_batch(self, batch):
        return batch["image"], batch["label"]

    def process_batch(self, batch):
        img, lab = self.unpack_batch(batch)
        out = self.forward(img)
        out = out.squeeze()
        if self.classification == "binary":
            prob = torch.sigmoid(out)
            if img.shape[0] == 1:
                prob = prob.unsqueeze(0)
            loss = F.binary_cross_entropy(prob, lab)
        else:
            raise NotImplementedError(
                f"Classification {self.classification} not implemented"
            )
        return loss

    def training_step(self, batch):
        loss = self.process_batch(batch)
        return loss

    def validation_step(self, batch):
        loss = self.process_batch(batch)
        return loss

    def test_step(self, batch):
        loss = self.process_batch(batch)
        return loss

def get_tv_model(
    model_name,
    num_classes=1,
    batch_size=64,
    d_size=500,
    pretrained=False,
    channel_size=3,
):
    model = TorchvisionModel(
        model_name=model_name,
        num_classes=num_classes,
        print_freq=250,
        batch_size=batch_size,
        d_size=d_size,
        pretrained=pretrained,
        channel_size=channel_size,
    )
    return model


def load_last_pt_ckpt(model_name ,ckpt_path, channel_size):
    if os.path.isfile(ckpt_path):
        if USE_GPU:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = torch.device("cpu")
        print(f"Loading checkpoint from {ckpt_path}")
        if "pth" in ckpt_path:
            checkpoint = torch.load(ckpt_path, map_location=device)
            state_dic = checkpoint["model_state_dict"]
            model = get_tv_model(model_name,channel_size=channel_size)
            model.load_state_dict(state_dic)
        elif "ckpt" in ckpt_path:
            model = get_tv_model(channel_size=channel_size)
            model = model.load_from_checkpoint(ckpt_path)
        return model
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    
def torch_prediction(model, data_loader, device):
    model.eval()
    model.to(device)
    with torch.no_grad():
        output_list = []
        patient_id_list = []
        hour_list = []
        quality_list = []
        for _, batch in enumerate(tqdm(data_loader)):
            data, features, targets, ids, hours, qualities = (
                batch["image"],
                batch["features"],
                batch["label"],
                batch["id"],
                batch["hour"],
                batch["quality"],
            )
            data = data.to(device)
            features = features.to(device)
            outputs = model(data, features)
            outputs = torch.sigmoid(outputs)
            output_list = output_list + outputs.cpu().numpy().tolist()
            patient_id_list = patient_id_list + ids
            hour_list = hour_list + list(hours.cpu().detach().numpy())
            quality_list = quality_list + list(qualities.cpu().detach().numpy())
    return output_list, patient_id_list, hour_list, quality_list


def get_challenge_models(model_folder, model_filename): 
    print('get challenge model')
    print(model_filename)
    print('model_folder: ',model_folder)
    model_folder = model_folder.lower()
    
    filename = hf_hub_download(repo_id=model_folder, filename="models.sav")
    model = joblib.load(filename)
    file_path_eeg = hf_hub_download(repo_id=model_folder, filename=model_filename+'.pth')
    if USE_TORCH:
        print("Load model...")
        model["torch_model_eeg"] = load_last_pt_ckpt(model_filename, 
            file_path_eeg, channel_size=len(EEG_CHANNELS)
        )
    else:
        model["torch_model_eeg"] = None
    return model

def train_challenge_model(data_folder, model_folder, patient_ids, verbose):
    params_torch = PARAMS_TORCH
    train_ids, val_ids = train_test_split(patient_ids, test_size=params_torch["val_size"], random_state=42)
    
    if USE_GPU:
        device = torch.device(
            "cuda:1"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device("cpu")
        
    if LIM_HOURS_DURING_TRAINING:
        hours_to_use = NUM_HOURS_EEG_TRAINING
    else:
        hours_to_use = None
    
    # Load multiple base models
    models = {
        "densenet121": get_challenge_models(model_folder, "densenet121")["torch_model_eeg"],
        "efficientnet_v2_s": get_challenge_models(model_folder, "efficientnet_v2_s")["torch_model_eeg"],
        "convnext_tiny": get_challenge_models(model_folder, "convnext_tiny")["torch_model_eeg"],
        # "resnet50": get_challenge_models(model_folder, "resnet50")["torch_model_eeg"]
    }
    
    # Create datasets
    train_dataset_eeg = RecordingsDataset(
        data_folder,
        patient_ids=train_ids,
        device=device,
        group="EEG",
        hours_to_use=hours_to_use,
    )
    val_dataset_eeg = RecordingsDataset(
        data_folder,
        patient_ids=val_ids,
        device=device,
        group="EEG",
        hours_to_use=hours_to_use,
    )
    
    # Create data loaders
    train_loader_eeg = DataLoader(
        train_dataset_eeg,
        batch_size=params_torch["batch_size"],
        num_workers=PARAMS_DEVICE["num_workers"],
        shuffle=True,
        pin_memory=True
    )
    val_loader_eeg = DataLoader(
        val_dataset_eeg,
        batch_size=params_torch["batch_size"],
        num_workers=PARAMS_DEVICE["num_workers"],
        shuffle=False,
        pin_memory=True
    )
    
    # Get predictions from each model for training set
    train_predictions = {}
    train_labels = []
    train_patient_ids = []
    
    for model_name, model in models.items():
        model.eval()
        model.to(device)
        predictions = []
        labels = []
        patient_ids = []
        
        with torch.no_grad():
            for batch in train_loader_eeg:
                data = batch["image"].to(device)
                target = batch["label"].cpu().numpy()
                pid = batch["id"]
                
                output = model(data)
                prob = output.cpu().numpy()
                
                predictions.extend(prob)
                labels.extend(target)
                patient_ids.extend(pid)
        
        train_predictions[model_name] = np.array(predictions)
        train_labels = np.array(labels)
        train_patient_ids = patient_ids
    
    # Get predictions from each model for validation set
    val_predictions = {}
    val_labels = []
    val_patient_ids = []
    
    for model_name, model in models.items():
        model.eval()
        model.to(device)
        predictions = []
        labels = []
        patient_ids = []
        
        with torch.no_grad():
            for batch in val_loader_eeg:
                data = batch["image"].to(device)
                target = batch["label"].cpu().numpy()
                pid = batch["id"]
                
                output = model(data)
                prob = output.cpu().numpy()
                
                predictions.extend(prob)
                labels.extend(target)
                patient_ids.extend(pid)
        
        val_predictions[model_name] = np.array(predictions)
        val_labels = np.array(labels)
        val_patient_ids = patient_ids
    
    # Prepare data for meta-model
    X_train = np.column_stack([train_predictions[model_name] for model_name in models.keys()])
    X_val = np.column_stack([val_predictions[model_name] for model_name in models.keys()])
    y_train = train_labels
    y_val = val_labels
    
    # Train meta-model (Random Forest)
    from sklearn.ensemble import RandomForestClassifier
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    print(X_train, y_train)
    meta_model.fit(X_train, y_train)
    
    # Evaluate meta-model
      # Predict on validation set
    y_val_pred = meta_model.predict(X_val)
    y_val_proba = meta_model.predict_proba(X_val)[:, 1]

    # Compute metrics
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)

    # Print metrics
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Validation F1 Score: {val_f1:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall: {val_recall:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")

    # Optionally save metrics to a file
    metrics_path = os.path.join(model_folder, "meta_model_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {val_acc:.4f}\n")
        f.write(f"F1 Score: {val_f1:.4f}\n")
        f.write(f"Precision: {val_precision:.4f}\n")
        f.write(f"Recall: {val_recall:.4f}\n")
        f.write(f"AUC: {val_auc:.4f}\n")
    print(f"Saved meta-model metrics to {metrics_path}")
    

    # Save meta-model
    save_path = os.path.join(model_folder, "meta_model.sav")
    joblib.dump(meta_model, save_path)
    print(f"Saved meta-model to {save_path}")
    
    return meta_model, models

      
def run_challenge_models(model_folder, data_folder, patient_id, verbose):
    # Load meta-model
    models = {
        "densenet121": get_challenge_models(model_folder, "densenet121")["torch_model_eeg"],
        "efficientnet_v2_s": get_challenge_models(model_folder, "efficientnet_v2_s")["torch_model_eeg"],
        "convnext_tiny": get_challenge_models(model_folder, "convnext_tiny")["torch_model_eeg"],
    }

    try:
        meta_model_path = hf_hub_download(repo_id=model_folder.lower(), filename="meta_model.sav")
        meta_model = joblib.load(meta_model_path)
    except Exception as e:
        raise FileNotFoundError(f"Meta model not found in repository {model_folder}") from e
    # meta_model_path = os.path.join(model_folder, "meta_model.sav")
    # if not os.path.exists(meta_model_path):
    #     raise FileNotFoundError(f"Meta model not found at {meta_model_path}")
    # meta_model = joblib.load(meta_model_path)

    # Load data
    if verbose >= 2:
        print("Loading data...")
    
    # Get predictions from each model
    all_outputs = {}
    all_patient_ids = {}
    all_hours = {}
    all_qualities = {}
    
    for model_name, model in models.items():
        print(f"Predicting with {model_name}...")
        if verbose >= 2:
            print(f"Predicting with {model_name}...")
            
        if USE_GPU:
            device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            device = torch.device("cpu")
            
        model.eval()
        model.to(device)
        
        data_set_eeg = RecordingsDataset(
            data_folder,
            patient_ids=[patient_id],
            device=device,
            load_labels=False,
            group="EEG",
            hours_to_use=NUM_HOURS_EEG,
        )
        
        data_loader_eeg = DataLoader(
            data_set_eeg,
            batch_size=1,
            num_workers=PARAMS_DEVICE["num_workers"],
            shuffle=False
        )
        
        # Get predictions from each model
        (
            output_list_eeg,
            patient_id_list_eeg,
            hour_list_eeg,
            quality_list_eeg,
        ) = torch_prediction(model, data_loader_eeg, device)
        
        all_outputs[model_name] = output_list_eeg
        all_patient_ids[model_name] = patient_id_list_eeg
        all_hours[model_name] = hour_list_eeg
        all_qualities[model_name] = quality_list_eeg
        
    X_test = np.column_stack([all_outputs[model_name] for model_name in models.keys()])
    X_test_pred_proba = meta_model.predict_proba(X_test)[:, 1]
    
    (
        outcome_probability, 
        outcome, 
        name,
        segment_outcomes
    ) = torch_predictions_for_patient(
        X_test_pred_proba,
        all_patient_ids[model_name],
        all_hours[model_name],
        all_qualities[model_name],
        patient_id,
        hours_to_use=NUM_HOURS_EEG,
        group="EEG",
    )
    print('Outcome: ', outcome)
    print('Outcome probability: ', outcome_probability)
    
    
    return outcome, outcome_probability, segment_outcomes 
    
def torch_predictions_for_patient(
    output_list,
    patient_id_list,
    hour_list,
    quality_list,
    patient_id,
    max_hours=72,
    min_quality=0,
    num_signals=None,
    hours_to_use=None,
    group="",
):
    

    # Aggregate the probabilities

    segment_outcomes = [1 if v > DECISION_THRESHOLD else 0 for v in output_list]
    total_votes = len(segment_outcomes)
    positive_votes = sum(segment_outcomes)
    outcome_probability = positive_votes / total_votes
   
    agg_outcome = 1 if outcome_probability > VOTING_POS_MAJORITY_THRESHOLD else 0
    count_aux = ["voted"]
        
    torch_names = [f"prob_{group}_torch_{i}" for i in count_aux]
    
    return (
        outcome_probability,
        agg_outcome, 
        torch_names,
        segment_outcomes
    )


