import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Helpers_General.Self_Supervised_learning_Helpers.SSL_Architectures import MAEModel
from Helpers_General.Self_Supervised_learning_Helpers.SSL_functions import patchify, PILImageDataset
from Helpers_General.LoadImages import load_images_from_folder


def train_mae(model, dataloader, loss_function, epochs=5, lr=1e-4, patch_size=16):
    # Select GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    epoch_losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs in dataloader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            gt_patches = patchify(imgs, patch_size=patch_size)
            loss = loss_function(outputs, gt_patches)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    return model, epoch_losses

# --- Running the Training ---
class LoadImages:
    pass


if __name__ == "__main__":
    # Create dataset and dataloader
    #dataset = DummyImageDataset(num_samples=2000)
    original_images = load_images_from_folder("D:\PHD\PhdData\SSL_DATA_PATCHES")
    dataset = PILImageDataset(original_images)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0) #This wraps an iterable over our dataset, and supports automatic batching, sampling, shuffling and multiprocess data loading.


    # Instantiate the MAE model
    model = MAEModel()
    #model = MaskedAutoencoderViT()
    print(model)

    loss_fn = nn.MSELoss()
    # Train the MAE model
    trained_model, losses = train_mae(model, dataloader, loss_fn)

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

stop = 1

"""After training a pretext model
From gpt: 

# Example of loading encoder weights into a detection backbone
pretrained_dict = torch.load("pretrained_mae_encoder.pth")
model.encoder.load_state_dict(pretrained_dict)
# Then integrate this encoder into your detection model.

"""

