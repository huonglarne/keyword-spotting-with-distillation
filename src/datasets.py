from pathlib import Path

import os

from torchaudio.datasets import SPEECHCOMMANDS
import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset

from src.constants import LABEL_LIST, NFFT, STANDARD_AUDIO_LENGTH, AUDIOS_PATH, DATA_PATH

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


def _load_precompute(filepath: Path) -> torch.Tensor:
    return torch.load(filepath)


def _preprocess_audio_input(filepath: Path) -> torch.Tensor:
    audio, _ = torchaudio.load(filepath)
    # audio = audio.squeeze(0)

    pad = torch.zeros(1, STANDARD_AUDIO_LENGTH - audio.shape[-1])
    audio = torch.cat((audio, pad), dim=1)
    return audio

    # transform = torchaudio.transforms.Spectrogram(n_fft=NFFT, normalized=True)
    # spec = transform(audio)
    return spec

def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


def label_to_index(word):
    # Return the position of the word in labels
    return torch.tensor(LABEL_LIST.index(word))


def simconv_collate_fn(batch):

    # A data tuple has the form:
    # waveform, sample_rate, label, speaker_id, utterance_number

    tensors, targets = [], []

    # Gather in lists, and encode labels as indices
    for waveform, _, label, *_ in batch:
        tensors += [waveform]
        targets += [label_to_index(label)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    targets = torch.stack(targets)

    return tensors, targets

class AudioDataset(SPEECHCOMMANDS):
    def __init__(self, data_dir: Path = DATA_PATH , subset: str = None):
        super().__init__(data_dir, download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        self._walker = load_list(f"{subset}_list.txt")


class AudioDistillDataset(AudioDataset):
    def __init__(self, audios_path: Path, teacher_preds_path: Path, subset: str):
        super().__init__(audios_path, subset)
        self.teacher_preds_path_list = [teacher_preds_path / sub_path.strip().replace('wav', 'pt') for sub_path in self.file_list]

    def __getitem__(self, index):
        student_input, label = super().__getitem__(index)
        
        teacher_preds = _load_precompute(self.teacher_preds_path_list[index])
        return student_input, teacher_preds, label