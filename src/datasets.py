from pathlib import Path
import torchaudio
import torch
import torch.nn.functional
from torch.utils.data import Dataset

from src.constants import LABEL_LIST, NFFT, STANDARD_AUDIO_LENGTH

def _create_train_subset_file(base_path, replace_existing=False):
    train_filepath = base_path / 'train_list.txt'

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
    audio = audio.squeeze(0)

    pad = torch.zeros(STANDARD_AUDIO_LENGTH - audio.shape[0])
    audio = torch.cat((audio, pad))

    transform = torchaudio.transforms.Spectrogram(n_fft=NFFT, normalized=True)
    spec = transform(audio)
    return spec


class AudioDataset(Dataset):
    def __init__(self, audios_path: Path, subset: str):
        if subset == "train":
            _create_train_subset_file(audios_path)

        with open(audios_path / f'{subset}_list.txt', 'r') as f:
            self.file_list = f.readlines()

        self.audio_path_list = [audios_path / sub_path.strip() for sub_path in self.file_list]

        label_list = [filepath.parent.stem for filepath in self.audio_path_list]
        self.label_list = [LABEL_LIST.index(label) for label in label_list]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        input = _preprocess_audio_input(self.audio_path_list[idx])
        label = self.label_list[idx]

        return {
            "input": input,
            "label": label
        }


class AudioDistillDataset(AudioDataset):
    def __init__(self, audios_path: Path, teacher_preds_path: Path, subset: str):
        super().__init__(audios_path, subset)
        self.teacher_preds_path_list = [teacher_preds_path / sub_path.strip().replace('wav', 'pt') for sub_path in self.file_list]

    def __getitem__(self, index):
        audio_dataset_output = super().__getitem__(index)
        student_input = audio_dataset_output['input']
        label = audio_dataset_output['label']

        teacher_preds = _load_precompute(self.teacher_preds_path_list[index])

        return {
            "student_input": student_input,
            "teacher_preds": teacher_preds,
            "label": label
        }
