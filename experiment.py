from src.datasets import AudioDataset, AudioDistillDataset, simconv_collate_fn
from torch.utils.data import DataLoader


subset = 'training'
audio_dataset = AudioDistillDataset(subset=subset)

batch_size = 3

data_loader = DataLoader(
    audio_dataset,
    batch_size=batch_size,
    num_workers=2,
    pin_memory=True,
    collate_fn=simconv_collate_fn
)

for input, label in data_loader:
    break