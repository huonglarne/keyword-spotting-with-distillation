import os

from torchaudio.datasets import SPEECHCOMMANDS
from datasets import create_train_subset_file

from src.constants import AUDIOS_PATH, DATA_PATH
from src.teacher_precompute import precompute_teacher_preds

os.environ['TORCH_HOME'] = 'checkpoints'

SPEECHCOMMANDS(DATA_PATH, download=True)

create_train_subset_file(AUDIOS_PATH, replace_existing=True)

precompute_teacher_preds(batch_size=1024)