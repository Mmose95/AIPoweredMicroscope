import torch


def DINO_training_loop(model, optimizer, criterion, num_epochs, dataloader, device):

    model.to(device)

    # Print model summary
    print(model)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for images in dataloader:
            images = images.to(device)

            optimizer.zero_grad()

            # Forward pass
            features = model(images)

            # Self-supervised objective (Here, we use MSE for simplicity)
            loss = criterion(features, features.clone().detach())

            # Backprop
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(dataloader)}")

    # Save the pretrained model
    torch.save(model.state_dict(), "dinov2_pretrained_microscopy.pth")
