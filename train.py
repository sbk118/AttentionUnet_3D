from tqdm import tqdm
import torch

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for images, masks in tqdm(dataloader, desc="🟢 Training", leave=False):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
