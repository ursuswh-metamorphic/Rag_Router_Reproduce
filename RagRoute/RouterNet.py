import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class RouterNet(nn.Module):
    def __init__(self, emb_dim=768):
        # MedCPT uses 768-dim query/article embeddings.
        # input_dim = Query(768) + Centroid(768) + Dist(1) + Size(1) + Density(1) = 1539
        input_total_dim = (emb_dim * 2) + 3
        super(RouterNet, self).__init__()
        
        # Layer 1: 256 neurons + LayerNorm + ReLU + Dropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Layer 2: 128 neurons + LayerNorm + ReLU + Dropout
        self.layer2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Output layer: 1 neuron (Raw logit)
        self.output_layer = nn.Linear(128, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.output_layer(x)

def train_router(model, train_loader, val_loader, epochs=50, device='cpu'):
    # Tính toán pos_weight: Giả sử nhãn 0 gấp 4 lần nhãn 1 (đặc thù của RAGRoute)
    # Trong thực tế, bạn nên tính tỷ lệ này từ tập train thực tế.
    pos_weight = torch.tensor([4.0]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Cyclic LR: Oscillating between 0.001 and 0.005 như bài báo
    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.005, 
                                            step_size_up=5, mode='triangular')

    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels.float())
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        
        # Validation logic (có thể thêm accuracy check ở đây)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

    return model