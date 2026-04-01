#!/usr/bin/env python
# coding: utf-8

# In[4]:


import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import kagglehub


# In[5]:


NUM_CLASSES = 10 
SEQUENCE_LENGTH = 16
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
IMAGE_SIZE = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Using device: {device}")


# In[6]:


class LeapGestureSequenceDataset(Dataset):
    def __init__(self, root_dir, seq_len=16, transform=None):
        """
        Groups individual image frames into temporal sequences for the CNN-LSTM.
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.transform = transform
        self.samples = []
        
      
        self.gesture_mapping = {
            '01_palm': 0, '02_l': 1, '03_fist': 2, '04_fist_moved': 3, 
            '05_thumb': 4, '06_index': 5, '07_ok': 6, '08_palm_moved': 7, 
            '09_c': 8, '10_down': 9
        }

        
        subjects = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        
        for subject in subjects:
            subject_path = os.path.join(root_dir, subject)
            gestures = [d for d in os.listdir(subject_path) if os.path.isdir(os.path.join(subject_path, d))]
            
            for gesture in gestures:
                if gesture not in self.gesture_mapping: 
                    continue
                    
                label = self.gesture_mapping[gesture]
                gesture_path = os.path.join(subject_path, gesture)
                
                
                images = sorted([f for f in os.listdir(gesture_path) if f.endswith('.png')])
                
               
                for i in range(0, len(images) - self.seq_len + 1, self.seq_len):
                    sequence_paths = [os.path.join(gesture_path, images[i+j]) for j in range(self.seq_len)]
                    self.samples.append((sequence_paths, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sequence_paths, label = self.samples[idx]
        video_tensor = []
        
        for img_path in sequence_paths:
           
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            video_tensor.append(image)
            
       
        video_tensor = torch.stack(video_tensor)
        return video_tensor, label


# In[7]:


class SpatialCNN(nn.Module):
    def __init__(self, embed_size):
        super(SpatialCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Flatten()
        )
        
        self.fc = nn.Linear(4096, embed_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.dropout(x)
        return self.fc(x)


class SmartHomeGesture_CNNLSTM(nn.Module):
    def __init__(self, num_classes, embed_size=256, hidden_size=512, num_layers=2):
        super(SmartHomeGesture_CNNLSTM, self).__init__()
        
        self.cnn = SpatialCNN(embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0.0
        )
        self.fc_out = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        batch_size, seq_len, C, H, W = x.size()
        
        
        c_in = x.view(batch_size * seq_len, C, H, W)
        c_out = self.cnn(c_in)
        
       
        r_in = c_out.view(batch_size, seq_len, -1)
        lstm_out, _ = self.lstm(r_in)
        
        
        final_time_step = lstm_out[:, -1, :]
        return self.fc_out(final_time_step)


# In[8]:


if __name__ == "__main__":
    print("[*] Downloading LeapGestRecog dataset via kagglehub...")
    dataset_path = kagglehub.dataset_download("gti-upm/leapgestrecog")
    root_data_dir = os.path.join(dataset_path, "leapGestRecog")
    print(f"[*] Dataset ready at: {root_data_dir}")

    print("[*] Preparing DataLoaders...")
    video_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = LeapGestureSequenceDataset(
        root_dir=root_data_dir, 
        seq_len=SEQUENCE_LENGTH, 
        transform=video_transforms
    )
    
    print(f"[*] Total video sequences generated: {len(train_dataset)}")

    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=0 
    )

    print("[*] Initializing CNN-LSTM Model...")
    model = SmartHomeGesture_CNNLSTM(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("[*] Starting Training Loop...")
    model.train()

    for epoch in range(EPOCHS):
        running_loss = 0.0
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            videos, labels = videos.to(device), labels.to(device)
            
           
            optimizer.zero_grad()
            outputs = model(videos)
            
           
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            
            if batch_idx % 20 == 0:
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        avg_loss = running_loss / len(train_loader)
        print(f"--- Epoch {epoch+1} Completed | Average Loss: {avg_loss:.4f} ---")
        
    print("[*] Training Complete! Assignment ready for submission.")


# In[9]:


from torch.utils.data import random_split, DataLoader


torch.save(model.state_dict(), "gesture_model_final.pth")
print("[*] Model weights saved safely as 'gesture_model_final.pth'")


def evaluate_accuracy(model, loader, device):
    model.eval() 
    correct = 0
    total = 0
    
    print(f"[*] Starting evaluation on {len(loader.dataset)} sequences...")
    
    with torch.no_grad(): 
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy


train_size = int(0.8 * len(train_dataset))
test_size = len(train_dataset) - train_size
_, test_subset = random_split(train_dataset, [train_size, test_size])


test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


final_accuracy = evaluate_accuracy(model, test_loader, device)

print(f"\n=========================================")
print(f" Final Model Accuracy: {final_accuracy:.2f}%")
print(f"=========================================")


# In[11]:


def calculate_accuracy(model, loader, device):
    model.eval() 
    correct = 0
    total = 0
    
    with torch.no_grad(): 
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return 100 * correct / total


# In[12]:


train_acc = calculate_accuracy(model, train_loader, device)
test_acc = calculate_accuracy(model, test_loader, device)

print(f"Final Training Accuracy: {train_acc:.2f}%")
print(f"Final Validation Accuracy: {test_acc:.2f}%")


# In[2]:


get_ipython().system('pip install kagglehub')


# In[ ]:




