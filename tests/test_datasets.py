from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from src.datasets import AudioDataset, AudioDistillDataset, _create_train_subset_file, _preprocess_audio_input
from src.constants import LABEL_LIST, NFFT, STANDARD_AUDIO_LENGTH

def test_create_train_subset_file(audios_path):
    _create_train_subset_file(audios_path, replace_existing=True)

    train_filepath = audios_path / 'train_list.txt'
    assert train_filepath.exists()

    with open(train_filepath, 'r') as f:
        file_list = f.readlines()

    label_list = set([file.split('/')[0] for file in file_list])
    assert label_list == set(LABEL_LIST)


def test_preprocess_audio_input(audios_path):
    audio_path = audios_path / 'backward/0a2b400e_nohash_0.wav'

    preprocessed_audio = _preprocess_audio_input(audio_path)
    assert preprocessed_audio.shape == (STANDARD_AUDIO_LENGTH,)


def test_audio_dataset(audios_path):
    subset = 'train'
    audio_dataset = AudioDataset(audios_path, subset)

    batch_size = 3

    data_loader = DataLoader(
        audio_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True
    )

    for batch in data_loader:
        assert batch['input'].shape == (batch_size, NFFT//2+1, np.ceil(STANDARD_AUDIO_LENGTH/NFFT*2))
        assert batch['label'].shape == (batch_size,)
        break


def test_audio_distill_dataset(audios_path, teacher_preds_path):
    subset = 'train'
    audio_distill_dataset = AudioDistillDataset(audios_path, teacher_preds_path, subset)

    batch_size = 3

    data_loader = DataLoader(
        audio_distill_dataset,
        batch_size=batch_size,
        num_workers=1,
        pin_memory=True
    )

    for batch in data_loader:
        assert batch['teacher_preds'].shape == (batch_size, len(LABEL_LIST))
        break


