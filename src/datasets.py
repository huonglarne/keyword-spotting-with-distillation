from ast import Tuple
from pathlib import Path

import os
from typing import Callable, Optional

from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import torch
import torch.nn.functional

from src.constants import LABEL_LIST, NFFT, STANDARD_AUDIO_LENGTH, AUDIOS_PATH, DATA_PATH, TEACHER_PREDS_PATH

def create_train_subset_file(base_path, replace_existing=False):
    train_filepath = base_path / 'training_list.txt'

    if not replace_existing and train_filepath.exists():
        return

    with open(base_path / 'validation_list.txt', 'r') as f:
        val_list = f.readlines()
    with open(base_path / 'testing_list.txt', 'r') as f:
        test_list = f.readlines()
    val_test_list = set(test_list+val_list)

    all_list = []
    for path in base_path.glob('*/'):
        if path.stem in LABEL_LIST:
            audio_files = list(path.glob('*.wav'))
            file_list = [f"{f.parent.stem}/{f.name}" for f in audio_files]
            all_list += file_list
        else:
            print

    training_list = [x for x in all_list if x not in val_test_list]
    with open(train_filepath, 'w') as f:
        for line in training_list:
            f.write(f"{line}\n")


def _load_precompute(filepath: torch.Tensor) -> torch.Tensor:
    return torch.load(filepath)


def _preprocess_audio_input(audio) -> torch.Tensor:
    pad = torch.zeros(1, STANDARD_AUDIO_LENGTH - audio.shape[-1])
    audio = torch.cat((audio, pad), dim=1)
    return audio

    # transform = torchaudio.transforms.Spectrogram(n_fft=NFFT, normalized=True)
    # spec = transform(audio)
    return spec


class AudioDataset(SPEECHCOMMANDS):
    def __init__(self, data_dir: Path = DATA_PATH , subset: str = None, preprocess_fn: Optional[Callable] =_preprocess_audio_input):
        super().__init__(data_dir, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [Path(os.path.join(self._path, line.strip())) for line in fileobj]

        self._walker = load_list(f"{subset}_list.txt")
        self.preprocess_fn = preprocess_fn


    def __getitem__(self, n: int):
        audio_array, sample_rate, label_str, speaker_id, label = super().__getitem__(n)

        if self.preprocess_fn is not None:
            audio_array = self.preprocess_fn(audio_array)

        return audio_array, sample_rate, label_str, speaker_id, label


class AudioDistillDataset(AudioDataset):
    def __init__(self, data_path: Path = DATA_PATH, teacher_preds_path: Path = TEACHER_PREDS_PATH, subset: str = None):
        super().__init__(data_path, subset)
        self.teacher_preds_path_list = [teacher_preds_path / sub_path.parent.name / sub_path.name.replace('wav', 'pt') for sub_path in self._walker]

    def __getitem__(self, index):
        student_input, _, _, _, label = super().__getitem__(index)
        
        teacher_preds = _load_precompute(self.teacher_preds_path_list[index])
        return student_input, teacher_preds, label