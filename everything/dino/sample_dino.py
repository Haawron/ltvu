from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from transformers import ViTModel, ViTConfig, ViTFeatureExtractor


# Load a pretrained ViT model from the Hugging Face library
model_name = 'google/vit-base-patch16-224'

# Load a feature extractor for preprocessing images
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# Define the transform for the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# Load CIFAR-10 dataset
train_dataset = CIFAR10(root='./data', train=True, transform=transform, download=True)
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9):
        super(DINOLoss, self).__init__()
        self.teacher_temp = teacher_temp
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        # Normalize the outputs
        student_out = F.log_softmax(student_output / self.student_temp, dim=-1)  # log(p1)
        teacher_out = F.softmax((teacher_output - self.center) / self.teacher_temp, dim=-1)  # p2

        # Compute the cross-entropy loss
        loss = torch.sum(-teacher_out * student_out, dim=-1).mean()  # H(p2, p1) = -p2 * log(p1)

        # Update the center
        self.update_center(teacher_output)

        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # EMA update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DINO(nn.Module):
    def __init__(self, student, teacher):
        super(DINO, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher.eval()  # Freeze teacher

    def forward(self, x):
        student_output = self.student(x)
        with torch.no_grad():
            teacher_output = self.teacher(x)
        return student_output, teacher_output


# Initialize models
device = 'cuda'
student_model = ViTModel.from_pretrained('google/vit-base-patch16-224', local_files_only=True).to(device)
teacher_model = ViTModel.from_pretrained('google/vit-base-patch16-224', local_files_only=True).to(device)
teacher_model.eval()  # Freeze teacher model parameters

# Initialize DINO loss
dino_loss_fn = DINOLoss(out_dim=768).to(device)

optimizer = torch.optim.Adam(student_model.parameters(), lr=1e-4)

num_epochs = 10
for epoch in range(num_epochs):
    student_model.train()
    for images, target in tqdm(train_dataloader):
        print(images.shape)
        images = images.to(device)
        student_output = student_model(images).last_hidden_state  # [B, 1(CLS)+P, D]
        with torch.no_grad():
            teacher_output = teacher_model(images).last_hidden_state  # [B, 1(CLS)+P, D]

        loss = dino_loss_fn(student_output, teacher_output)

        tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update teacher model weights with EMA
        for param_student, param_teacher in zip(student_model.parameters(), teacher_model.parameters()):
            param_teacher.data = param_teacher.data * 0.99 + param_student.data * 0.01

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
