from pathlib import Path

from src.loss import student_loss, distillation_loss
from tests.conftest import audios_path, teacher_preds_path

from torch.utils.data import DataLoader
import torch
import torchvision

from src.datasets import AudioDistillDataset


audios_path = Path('data/speech_commands_v0.02')
teacher_preds_path = Path('data/teacher_preds')

loss = student_loss

model = torchvision.models.efficientnet_v2_s()

num_epochs = 2
batch_size = 3
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
    for i, data in enumerate(train_loader):
        # Forward pass: Compute predicted y by passing x to the model
        student_pred = model(data['student_input'])

        # Compute and print loss
        loss = loss(student_pred, data['teacher_pred'], data['label'])
        print(f'Epoch: {epoch+1}, Iteration: {i+1}, Loss: {loss.item()}')

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
