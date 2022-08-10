import os
from torch.utils.data import DataLoader
from src.teacher_precompute import get_pretrained_ast, _prepare_dataset_json
from src.ast.src.dataloader import AudiosetDataset
from src.constants import INFERENCE_AUDIO_CONFIG, LABEL_CSV

os.environ['TORCH_HOME'] = 'checkpoints'

def test_get_pretrained_ast():
    pretrained_mdl_path = 'checkpoints/speechcommands_10_10_0.9812.pth'    
    audio_model, device = get_pretrained_ast(pretrained_mdl_path)
    assert audio_model is not None


def test_prepare_dataset_json(audios_path):
    dataset_json_file = _prepare_dataset_json(audios_path, 'testing')

    dataset = AudiosetDataset(dataset_json_file, label_csv=LABEL_CSV, audio_conf=INFERENCE_AUDIO_CONFIG)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    for audio_inputs, labels in dataloader:
        assert audio_inputs is not None