from re import X
from pathlib import Path

import pytest

@pytest.fixture
def audios_path():
   return Path('keyword_spot_distill/data/speech_commands_v0.02')

@pytest.fixture
def teacher_preds_path():
   return Path('keyword_spot_distill/data/teacher_preds')