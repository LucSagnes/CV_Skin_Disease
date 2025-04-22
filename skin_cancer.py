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

    from models import SmallNetwork

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
def _(import_and_preprocess):
    import_and_preprocess(dataset="marmal88/skin_cancer",
                          resize=(256, 256),
                          centercrop=(224, 224),
                          batch_size=64,
                          shuffle=True)
    return


@app.cell
def _(load_dataset):
    ds = load_dataset("marmal88/skin_cancer")
    return (ds,)


@app.cell
def _():
    return


@app.cell
def _(ds, np, preprocess_transforms, torch):
    # Transforms train images in an array, into another array
    train_orig = torch.from_numpy(
        np.array([preprocess_transforms(ds['train'][idx]['image']) for idx in range(len(ds['train']))])
    )

    valid_orig = np.array([np.array(ds['validation'][idx]['image']) for idx in range(len(ds['validation']))])
    test_orig = np.array([np.array(ds['test'][idx]['image']) for idx in range(len(ds['test']))])
    return test_orig, train_orig, valid_orig


@app.cell
def _(LabelEncoder, ds, np):
    le_labels = LabelEncoder()

    train_labels = le_labels.fit_transform(np.array(ds['train']['dx']))
    # valid_labels = le_labels.transform(np.array(ds['validation']['dx']))
    # test_labels = le_labels.transform(np.array(ds['test']['dx']))
    return le_labels, train_labels


@app.cell
def _(train_orig):
    train_orig.shape
    return


@app.cell
def _(le_labels):
    dict(zip(le_labels.classes_, le_labels.transform(le_labels.classes_)))
    return


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    return (device,)


@app.cell
def _(DataLoader, TensorDataset, np, torch):
    def numpy_to_dataloader(data: np.array = None, labels: np.array = None, batch_size: int = 32):
        """
        Convert a numpy array to a fully prepared DataLoader object.

        Parameters
        ----------
        data : np.array
            Data as a numpy array
        labels : np.array
            Labels corresponding to the data, must be shape (n, ) or (n, 1)
        """
        # transform to a tensor and normalize
        torch_data = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
        torch_label = torch.from_numpy(labels).long()

        dataset = TensorDataset(torch_data, torch_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
    return (numpy_to_dataloader,)


@app.cell
def _(
    numpy_to_dataloader,
    test_labels,
    test_orig,
    train_labels,
    train_orig,
    valid_labels,
    valid_orig,
):
    batch_size = 128

    train_loader = numpy_to_dataloader(train_orig, train_labels, batch_size)
    valid_loader = numpy_to_dataloader(valid_orig, valid_labels, batch_size)
    test_loader = numpy_to_dataloader(test_orig, test_labels, batch_size)
    return batch_size, test_loader, train_loader, valid_loader


@app.cell
def _(display, train_orig, transforms):
    display(transforms.ToPILImage()(train_orig[24]))
    return


@app.cell
def _(display, ds):
    # resize = Resize(size=(256, 256))
    out = ds['train'][24]['image']
    display(out)
    return (out,)


@app.cell
def _(train_loader):
    train_loader.dataset[0]
    return


@app.cell
def _(resize, train_loader):
    resize(train_loader.dataset[2][0]).shape
    return


@app.cell
def _(F, SmallNetwork, device, optim, test_loader, torch, train_loader):
    model = SmallNetwork().to(device)

    # train the model
    n_epochs = 20
    for epoch in range(1, n_epochs + 1):
        # model.train() useful when there is dropout and BatchNorm
        for batch_idx, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            y_pred = model(data)
            loss = F.nll_loss(y_pred, label)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print("Train Epoch {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100 * batch_idx / len(train_loader), loss.item()
                ))

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                y_pred = model(data)
                test_loss += F.nll_loss(y_pred, label, reduction='sum').item()
                pred = y_pred.argmax(dim=1, keepdim=True)
                correct += pred.eq(label.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print("\nTest set : Average Loss: {:.4f}, Accuracy : {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset),
            100 * correct / len(test_loader.dataset)
        ))
    return (
        batch_idx,
        correct,
        data,
        epoch,
        label,
        loss,
        model,
        n_epochs,
        optimizer,
        pred,
        test_loss,
        y_pred,
    )


if __name__ == "__main__":
    app.run()
