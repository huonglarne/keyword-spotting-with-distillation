import os, sys
from pathlib import Path

import torch

from ast.src.models import ASTModel
from ast.src.data import AudiosetDataset

def get_pretrained_ast(pretrained_mdl_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(pretrained_mdl_path, map_location=device)
    audio_model = models.ASTModel(label_dim=35, fstride=10, tstride=10, input_fdim=128,
                                    input_tdim=128, imagenet_pretrain=True,
                                    audioset_pretrain=False, model_size='base384')

    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    return audio_model


class PrecomputeDataset(AudiosetDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index):
        fbank, label_indices = super().__getitem__(index)
        filepath = self.data[index]['wav']
        return fbank, label_indices, filepath


## ===========================================================
## Create dataloader for all data
## ===========================================================

dataset_json = './egs/speechcommands/data/datafiles/speechcommand_all_data.json'
label_csv = './egs/speechcommands/data/speechcommands_class_labels_indices.csv'
batch_size = 128
num_workers = 2

norm_stats = {'audioset':[-4.2677393, 4.5689974], 'esc50':[-6.6268077, 5.358466], 'speechcommands':[-6.845978, 5.5654526]}
target_length = {'audioset':1024, 'esc50':512, 'speechcommands':128}
noise = {'audioset': False, 'esc50': False, 'speechcommands':True}
audio_conf = {'num_mel_bins': 128, 'target_length': 128, 'freqm': 48, 'timem': 48, 'mixup': 0.6, 'dataset': 'speechcommands', 'mode':'train', 'mean': -6.845978, 'std': 5.5654526,
                  'noise':True}

val_audio_conf = {'num_mel_bins': 128, 'target_length': 128, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'speechcommands', 'mode':'evaluation', 'mean': -6.845978, 'std': 5.5654526, 'noise': False}


eval_loader = torch.utils.data.DataLoader(
        AudiosetDataset(dataset_json, label_csv=label_csv, audio_conf=val_audio_conf),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

## ===========================================================
## Precompute all data
## ===========================================================

output_dir = Path('./egs/speechcommands/data/precomputed_logits')

with torch.no_grad():
        for i, (audio_input, labels, filepaths) in enumerate(eval_loader):
                print(f'batch {i}')

                audio_input = audio_input.to(device)
                audio_output = audio_model(audio_input)

                predictions = audio_output.to('cpu').detach()

                for logit, path in zip(predictions, filepaths):
                        path = Path(path)
                        
                        filename = output_dir.joinpath(path.parent.stem)
                        filename.mkdir(parents=True, exist_ok=True)

                        filename = filename.joinpath(path.stem + '.pt')
                        torch.save(logit, filename)