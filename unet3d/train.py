from pathlib import Path
from sklearn.metrics import jaccard_score
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


from unet3d.collate import pad_collate
from unet3d.dataset import BaselineDataset
from unet3d.unet3d import UNet


def print_iou_per_class(
    targets: torch.Tensor,
    preds: torch.Tensor,
    nb_classes: int,
) -> None:
    """
    Compute IoU between predictions and targets, for each class.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
        nb_classes (int): Number of classes in the segmentation task.
    """

    # Compute IoU for each class
    # Note: I use this for loop to iterate also on classes not in the demo batch

    iou_per_class = []
    for class_id in range(nb_classes):
        iou = jaccard_score(
            targets == class_id,
            preds == class_id,
            average="binary",
            zero_division=0,
        )
        iou_per_class.append(iou)

    for class_id, iou in enumerate(iou_per_class):
        print(
            "class {} - IoU: {:.4f} - targets: {} - preds: {}".format(
                class_id, iou, (targets == class_id).sum(), (preds == class_id).sum()
            )
        )


def print_mean_iou(targets: torch.Tensor, preds: torch.Tensor) -> None:
    """
    Compute mean IoU between predictions and targets.

    Args:
        targets (torch.Tensor): Ground truth of shape (B, H, W).
        preds (torch.Tensor): Model predictions of shape (B, nb_classes, H, W).
    """

    mean_iou = jaccard_score(targets, preds, average="macro")
    print(f"meanIOU (over existing classes in targets): {mean_iou:.4f}")


def train_model(
    data_folder: Path,
    nb_classes: int,
    input_channels: int,
    num_epochs: int = 10,
    batch_size: int = 4,
    learning_rate: float = 1e-3,
    device: str = "cpu",
    verbose: bool = False,
) -> UNet:
    """
    Training pipeline.
    """
    # Create data loader
    dataset = BaselineDataset(data_folder)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=pad_collate, shuffle=True
    )

    # Initialize the model, loss function, and optimizer
    model = UNet(input_channels, nb_classes,dim=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Move the model to the appropriate device (GPU if available)
    device = torch.device(device)
    model.to(device)

    # Training loop
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        for i, (inputs, targets) in tqdm(enumerate(dataloader), total=len(dataloader)):
            # Move data to device
            inputs["S2"] = inputs["S2"].to(device)  # Satellite data
            targets = targets.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs["S2"],1) 
            outputs_median_time = torch.median(outputs,2).values

            # Loss computation
            loss = criterion(outputs_median_time, targets)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Get the predicted class per pixel (B, H, W)
            preds = torch.argmax(outputs_median_time, dim=1)

            # Move data from GPU/Metal to CPU
            targets = targets.cpu().numpy().flatten()
            preds = preds.cpu().numpy().flatten()

            if verbose:
                # Print IOU for debugging
                print_iou_per_class(targets, preds, nb_classes)
                print_mean_iou(targets, preds)

        # Print the loss for this epoch
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    print("Training complete.")
    return model


if __name__ == "__main__":
    # Example usage:
    model = train_model(
        data_folder=Path(
            "/Users/louis.stefanuto.c/Documents/pastis-benchmark-mines2024/DATA/TRAIN/"
        ),
        nb_classes=20,
        input_channels=10,
        num_epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        device="mps",
        verbose=True,
    )
