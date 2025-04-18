import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from IPython.display import display
    import pandas as pd
    import numpy as np

    from sklearn.model_selection import train_test_split

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    from models import SmallNetwork

    import matplotlib.pyplot as plt
    from PIL import Image
    return (
        DataLoader,
        F,
        Image,
        SmallNetwork,
        TensorDataset,
        display,
        mo,
        nn,
        np,
        optim,
        pd,
        plt,
        torch,
        train_test_split,
    )


@app.cell
def _(torch):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")
    return (device,)


@app.cell
def _(DataLoader, TensorDataset, np, pd, torch):
    def import_and_shape(data_path: str = None, as_array: bool = False) -> tuple:
        """
        Import csv data and puts it in the right format (here 28x28x3)

        Parameters
        ----------
        data_path : str
            Data path, as a string
        as_array : bool
            Return data and label as array

        Returns
        -------
        tuple

        Example
        -------
        >>> import_and_shape("Data/my_data.csv") -> data, label
        """

        tmp = pd.read_csv(data_path)
        data, label = tmp[[col for col in tmp.columns if "label" not in col]], tmp["label"]

        if as_array:
            data = np.reshape(data, (len(data), 28, 28, 3))
            label = np.array(label)

        return data, label


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
    return import_and_shape, numpy_to_dataloader


@app.cell
def _(import_and_shape, numpy_to_dataloader, train_test_split):
    data_path = "Data/hmnist_28_28_RGB.csv"
    X_orig, y_orig = import_and_shape(data_path=data_path, as_array=True)
    X_train, X_test, y_train, y_test = train_test_split(X_orig, y_orig, test_size=0.2)

    batch_size = 128

    train_loader = numpy_to_dataloader(X_train, y_train, batch_size)
    test_loader = numpy_to_dataloader(X_test, y_test, batch_size)
    return (
        X_orig,
        X_test,
        X_train,
        batch_size,
        data_path,
        test_loader,
        train_loader,
        y_orig,
        y_test,
        y_train,
    )


@app.cell
def _(Image, X_orig, display, np):
    image = Image.fromarray(X_orig[5].astype(np.uint8))
    display(image)
    return (image,)


@app.cell
def _():
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


@app.cell
def _(label):
    label.shape
    return


if __name__ == "__main__":
    app.run()
