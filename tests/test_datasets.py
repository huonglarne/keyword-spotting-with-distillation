from email.mime import audio
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
import torchaudio

from src.data.datasets import AudioDataset, AudioDistillDataset, create_train_subset_file, _preprocess_audio_input
from src.constants import AUDIOS_PATH, LABEL_LIST, NFFT, STANDARD_AUDIO_LENGTH, TEACHER_PREDS_PATH

def test_create_train_subset_file():
    audios_path = AUDIOS_PATH
    create_train_subset_file(audios_path, replace_existing=True)

    train_filepath = audios_path / 'training_list.txt'
    assert train_filepath.exists()

    with open(train_filepath, 'r') as f:
        file_list = f.readlines()

    label_list = set([file.split('/')[0] for file in file_list])
    assert label_list == set(LABEL_LIST)


def test_preprocess_audio_input():
    audios_path = AUDIOS_PATH
    audio_path = audios_path / 'backward/0a2b400e_nohash_0.wav'
    audio = torchaudio.load(audio_path)
    preprocessed_audio = _preprocess_audio_input(audio)
    assert preprocessed_audio.shape == (1, STANDARD_AUDIO_LENGTH)
    # assert preprocessed_audio.shape == (1, NFFT//2+1, np.ceil(STANDARD_AUDIO_LENGTH/NFFT*2))


def test_audio_dataset():
    subset = 'training'
    audio_dataset = AudioDataset(subset=subset)

    batch_size = 3

    data_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        num_workers=2,
        pin_memory=True
    )

    for audio_array, sample_rate, label_str, speaker_id, label in data_loader:
        assert audio_array.shape[0] == batch_size
        assert label.shape == (batch_size,)
        break


def test_audio_distill_dataset():
    audios_path = AUDIOS_PATH
    teacher_preds_path = TEACHER_PREDS_PATH
    subset = 'train'
    audio_distill_dataset = AudioDistillDataset(audios_path, teacher_preds_path, subset)

    batch_size = 3

    data_loader = DataLoader(
        audio_distill_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True
    )

    for student_input, teacher_preds, label in data_loader:
        assert teacher_preds.shape == (batch_size, len(LABEL_LIST))
        break


