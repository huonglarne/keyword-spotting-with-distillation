from pathlib import Path
from src.constants import LABEL_LIST


from src.loss import distillation_loss
from tests.conftest import audios_path, teacher_preds_path

from torch.utils.data import DataLoader
import torch
import torchvision

from src.datasets import AudioDistillDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audios_path = Path('data/speech_commands_v0.02')
teacher_preds_path = Path('data/teacher_preds')

criterion = distillation_loss

model = torchvision.models.mobilenet_v3_small(num_classes=len(LABEL_LIST))
model.to(device)

num_epochs = 2
batch_size = 512
optimizer = torch.optim.SGD(model.parameters(), lr=1e-6)

audio_distill_dataset = AudioDistillDataset(audios_path, teacher_preds_path, 'train')
train_loader = DataLoader(
    audio_distill_dataset,
    batch_size=batch_size,
    num_workers=1,
    pin_memory=True
)

num_epochs = 2
for epoch in range(num_epochs):
    for i, (input, teacher_preds, label) in enumerate(train_loader):
        input = input.repeat(1, 3, 1, 1)
        input = input.to(device)
        teacher_preds = teacher_preds.to(device)
        label = label.to(device)

        student_pred = model(input)

        # Compute and print loss
        loss = criterion(student_pred, teacher_preds, label)
        print(f'Epoch: {epoch+1}, Iteration: {i+1}, Loss: {loss.item()}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
