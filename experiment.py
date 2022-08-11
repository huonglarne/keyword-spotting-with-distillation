from pathlib import Path
from src.constants import AUDIOS_PATH
from tqdm import tqdm

from torch.utils.data import DataLoader

from src.datasets import AudioDataset, AudioDistillDataset, simconv_collate_fn

subset = 'training'
# audio_dataset = AudioDataset(AUDIOS_PATH, subset)

audio_dataset = AudioDataset(Path('data'), subset)

batch_size = 256

data_loader = DataLoader(
    audio_dataset,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    collate_fn=simconv_collate_fn
)

for i, (data, label) in tqdm(enumerate(data_loader)):
    print(data.shape)
    if i>200:
        break