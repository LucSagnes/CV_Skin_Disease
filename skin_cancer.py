import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    from typing import Union

    import marimo as mo

    from IPython.display import display
    import pandas as pd
    import numpy as np

    from sklearn.preprocessing import LabelEncoder

    import torch
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from torchvision.transforms import transforms

    from models import SmallNetwork, ResNet18

    from preprocessing import import_and_preprocess

    import matplotlib.pyplot as plt
    from PIL import Image

    from datasets import load_dataset
    return (
        DataLoader,
        Dataset,
        F,
        Image,
        LabelEncoder,
        ResNet18,
        SmallNetwork,
        TensorDataset,
        Union,
        display,
        import_and_preprocess,
        load_dataset,
        mo,
        nn,
        np,
        optim,
        pd,
        plt,
        torch,
        transforms,
    )


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    return (device,)


@app.cell
def _(import_and_preprocess):
    train_dataloader, valid_dataloader, test_dataloader, label_mapping = import_and_preprocess(
        dataset="marmal88/skin_cancer",
        resize=(256, 256),
        centercrop=(224, 224),
        batch_size=64,
        shuffle=True
    )
    return label_mapping, test_dataloader, train_dataloader, valid_dataloader


@app.cell
def _(display, label_mapping, np, train_dataloader, transforms):
    random_idx = np.random.choice([i for i in range(len(train_dataloader))])
    img = transforms.ToPILImage()(train_dataloader.dataset[random_idx][0])
    img_label = label_mapping[int(train_dataloader.dataset[random_idx][1])]
    print(img_label)
    display(img)
    return img, img_label, random_idx


@app.cell
def _(ResNet18, device, nn, optim, torch, train_dataloader, valid_dataloader):
    # model = SmallNetwork().to(device)
    model = ResNet18(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # train the model for n epochs
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        train_loss_list = []
        train_correct = 0
        model.train() # Set model in training mode (useful for BatchNorm and Dropout)

        for batch_idx, (data, label) in enumerate(train_dataloader):
            data, label = data.to(device), label.to(device)

            # Make predictions and compute loss
            y_pred = model(data)
            loss = criterion(y_pred, label)
            train_loss_list.append(loss)

            # One step optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Retrieve correct predictions and count them
            train_pred = y_pred.argmax(dim=1, keepdim=True)
            train_correct += train_pred.eq(label.view_as(train_pred)).sum().item()

            # Show metrics foreach 10 batch
            if batch_idx % 10 == 0:
                print("Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_dataloader.dataset),
                    100 * batch_idx / len(train_dataloader), loss.item()
                ))

        # Average loss and accuracy for the training set
        avg_train_loss = sum(train_loss_list) / len(train_loss_list)
        train_size = len(train_dataloader.dataset)
        print("\nTrain set : Average Loss : {}, Accuracy : {}/{} ({:.0f}%)".format(
            avg_train_loss, train_correct, train_size, train_correct / train_size * 100
        ))

        # Validation part
        val_loss_list = [] 
        val_correct = 0
        model.eval() # Set the model in evaluation mode
        with torch.no_grad():
            for data, label in valid_dataloader:
                data, label = data.to(device), label.to(device)

                # Predictions and loss calculation
                y_pred = model(data)
                val_loss = criterion(y_pred, label)
                val_loss_list.append(val_loss)

                # Count correct predictions
                pred = y_pred.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(label.view_as(pred)).sum().item()

        # Average loss and accuracy for validation set
        avg_val_loss = sum(val_loss_list) / len(val_loss_list)
        val_size = len(valid_dataloader.dataset)
        print("\nValidation set : Average Loss: {:.4f}, Accuracy : {}/{} ({:.0f}%)\n".format(
            avg_val_loss, val_correct, val_size, 100 * val_correct / val_size
        ))
    return (
        avg_train_loss,
        avg_val_loss,
        batch_idx,
        criterion,
        data,
        epoch,
        label,
        loss,
        model,
        n_epochs,
        optimizer,
        pred,
        train_correct,
        train_loss_list,
        train_pred,
        train_size,
        val_correct,
        val_loss,
        val_loss_list,
        val_size,
        y_pred,
    )


if __name__ == "__main__":
    app.run()
