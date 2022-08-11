from pathlib import Path

STANDARD_AUDIO_LENGTH = 16000

LABEL_LIST =  ["backward", "follow", "five", "bed", "zero", "on", "learn", "two", "house", "tree", "dog", "stop", "seven", "eight", "down", "six", "forward", "cat", "right", "visual", "four",
    "wow", "no", "nine", "off", "three", "left", "marvin", "yes", "up", "sheila", "happy", "bird", "go", "one"
]

NFFT = 1024

AUDIOS_PATH = Path('data/SpeechCommands/speech_commands_v0.02')
DATA_PATH = Path('data/')

TEACHER_PREDS_PATH = Path('data/teacher_preds')

# for AST model
AST_MODEL_PATH = Path('checkpoints/speechcommands_10_10_0.9812.pth')
INFERENCE_AUDIO_CONFIG = {'num_mel_bins': 128, 'target_length': 128, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'speechcommands', 'mode':'evaluation', 'mean': -6.845978, 'std': 5.5654526, 'noise': False}
LABEL_CSV = 'src/ast/egs/speechcommands/data/speechcommands_class_labels_indices.csv'

