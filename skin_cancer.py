import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import matplotlib.pyplot as plt
    return DataLoader, F, TensorDataset, mo, nn, np, optim, pd, plt, torch


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
        torch_data = torch.from_numpy(data).permute(0, 3, 1, 2).float() / 255.0
        torch_label = torch.from_numpy(labels).long()

        dataset = TensorDataset(torch_data, torch_label)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return dataloader
    return import_and_shape, numpy_to_dataloader


@app.cell
def _(import_and_shape, numpy_to_dataloader):
    data_path = "Data/hmnist_28_28_RGB.csv"
    X_orig, y_orig = import_and_shape(data_path=data_path, as_array=True)

    train_loader = numpy_to_dataloader(X_orig, y_orig, 128)
    return X_orig, data_path, train_loader, y_orig


@app.cell
def _(F, nn, torch):
    class SmallNetwork(nn.Module):

        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), stride=1, padding=0)
            self.act1 = nn.ReLU()

            self.conv2 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5, 5), stride=1, padding=0)
            self.act2 = nn.ReLU()
        
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=1, padding=0)
        
            self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=1, padding=0)
            self.act3 = nn.ReLU()

            self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)

            self.flat = nn.Flatten()
        
            self.fc1 = nn.Linear(in_features=10_368, out_features=6_400)
            self.fc2 = nn.Linear(in_features=6_400, out_features=1_280)
            self.fc3 = nn.Linear(in_features=1_280, out_features=7)

        def forward(self, x):
            x = self.conv1(x)
            x = self.act1(x)

            x = self.conv2(x)
            x = self.act2(x)
        
            x = self.pool1(x)

            x = self.conv3(x)
            x = self.act3(x)

            x = self.pool2(x)

            x = torch.flatten(x, 1)
        
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.fc3(x)
            output = F.log_softmax(x, dim=1)

            return output
    return (SmallNetwork,)


@app.cell
def _(F, SmallNetwork, device, optim, train_loader):
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
    return (
        batch_idx,
        data,
        epoch,
        label,
        loss,
        model,
        n_epochs,
        optimizer,
        y_pred,
    )


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
