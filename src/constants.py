STANDARD_AUDIO_LENGTH = 16000

LABEL_LIST =  ["backward", "follow", "five", "bed", "zero", "on", "learn", "two", "house", "tree", "dog", "stop", "seven", "eight", "down", "six", "forward", "cat", "right", "visual", "four",
    "wow", "no", "nine", "off", "three", "left", "marvin", "yes", "up", "sheila", "happy", "bird", "go", "one"
]

# for AST model
INFERENCE_AUDIO_CONFIG = {'num_mel_bins': 128, 'target_length': 128, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'speechcommands', 'mode':'evaluation', 'mean': -6.845978, 'std': 5.5654526, 'noise': False}
LABEL_CSV = 'src/ast/egs/speechcommands/data/speechcommands_class_labels_indices.csv'