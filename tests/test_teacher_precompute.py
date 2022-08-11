from ast import AST
import os
from pathlib import Path
from torch.utils.data import DataLoader
from src.data_utils.teacher_precompute import PrecomputeDataset, get_pretrained_ast, _prepare_dataset_json, _precompute_batch, precompute_teacher_preds
from src.ast.src.dataloader import AudiosetDataset
from src.constants import AST_MODEL_PATH, INFERENCE_AUDIO_CONFIG, LABEL_CSV, LABEL_LIST

os.environ['TORCH_HOME'] = 'checkpoints'

def test_get_pretrained_ast():
    pretrained_mdl_path = AST_MODEL_PATH 
    audio_model, _ = get_pretrained_ast(pretrained_mdl_path)
    assert audio_model is not None


def test_prepare_dataset_json(audios_path):
    dataset_json_file = _prepare_dataset_json(audios_path, 'testing')

    dataset = AudiosetDataset(dataset_json_file, label_csv=LABEL_CSV, audio_conf=INFERENCE_AUDIO_CONFIG)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=1)

    for audio_inputs, labels in dataloader:
        assert audio_inputs is not None


def test_precompute(audios_path):
    pretrained_mdl_path = AST_MODEL_PATH
    audio_model, device = get_pretrained_ast(pretrained_mdl_path)

    batch_size = 2
    dataset = PrecomputeDataset(audios_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=1)

    for audio_input, _, _ in dataloader:
        predictions = _precompute_batch(audio_model, audio_input, device)
        assert predictions.shape == (batch_size, len(LABEL_LIST))
        break