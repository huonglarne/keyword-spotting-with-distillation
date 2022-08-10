from pathlib import Path

import json

import torch
from torch.utils.data import DataLoader

from src.ast.src.models import ASTModel
from src.ast.src.dataloader import AudiosetDataset
from src.constants import LABEL_LIST, INFERENCE_AUDIO_CONFIG, LABEL_CSV

def get_pretrained_ast(pretrained_mdl_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sd = torch.load(pretrained_mdl_path, map_location=device)
    audio_model = ASTModel(label_dim=35, fstride=10, tstride=10, input_fdim=128,
                                    input_tdim=128, imagenet_pretrain=True,
                                    audioset_pretrain=False, model_size='base384')

    audio_model = torch.nn.DataParallel(audio_model)
    audio_model.load_state_dict(sd, strict=False)

    return audio_model, device


def _prepare_dataset_json(basepath: Path, subset: str):
    if subset == 'all':
        filelist = []
        for each in ['train', 'testing', 'validation']:
            with open(basepath / f'{each}_list.txt', 'r') as f:
                filelist += f.readlines()
    else:
        with open(basepath / f'{subset}_list.txt', 'r') as f:
            filelist = f.readlines()

    wav_list = []
    for file in filelist:
        label = file.split('/')[0]
        label = LABEL_LIST.index(label)

        filepath = basepath / file.strip()

        cur_dict = {"wav": str(filepath), "labels": '/m/spcmd'+str(label).zfill(2)}
        wav_list.append(cur_dict)

    outpath = basepath / f'{subset}_data.json'
    with open(outpath, 'w') as f:
        json.dump({'data': wav_list}, f, indent=1)
            
    return outpath
        

class PrecomputeDataset(AudiosetDataset):
    def __init__(self, dataset_path):
        dataset_json_file = _prepare_dataset_json(dataset_path, 'all')
        super().__init__(dataset_json_file, INFERENCE_AUDIO_CONFIG, LABEL_CSV)

    def __getitem__(self, index):
        fbank, label_indices = super().__getitem__(index)
        filepath = self.data[index]['wav']
        return fbank, label_indices, filepath


def precompute_batch(audio_model, audio_input, device):
    with torch.no_grad():
        audio_input = audio_input.to(device)
        audio_output = audio_model(audio_input)

        predictions = audio_output.to('cpu').detach()
    return predictions


def save_precompute(predictions, filepaths, output_dir):
    for logit, path in zip(predictions, filepaths):
        path = Path(path)
        
        filename = output_dir.joinpath(path.parent.stem)
        filename.mkdir(parents=True, exist_ok=True)

        filename = filename.joinpath(path.stem + '.pt')
        torch.save(logit, filename)


def precompute_teacher_preds(pretrained_mdl_path, audios_path, output_dir, batch_size=128, num_workers=2):
    audio_model, device = get_pretrained_ast(pretrained_mdl_path)

    dataset = PrecomputeDataset(audios_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    for audio_input, _, filepaths in dataloader:
        predictions = precompute_batch(audio_model, audio_input, device)
        save_precompute(predictions, filepaths, output_dir)

