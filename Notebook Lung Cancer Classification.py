import marimo

__generated_with = "0.12.8"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Import Libraries""")
    return


@app.cell
def _():
    import os 
    import pandas as pd
    import marimo as mo
    from IPython.display import display
    return display, mo, os, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        # Import the Data

        Data should be stored into a file Data where we have unzip the dataset from https://huggingface.co/datasets/Kabil007/LungCancer4Types/blob/main/Data-20240107T052410Z-001.zip

        The data in this dataset is organized in different file : a train, a test and a valid file. 

        In every file we can find images from 4 different classifications  of lung cancer : normal, ADC, Large ADC or squamous

        We will read from every file the images and we will label every images regarding their classes
        """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""# Functions""")
    return


@app.cell
def _(os, pd):

    def create_image_dataframe(data_path, extensions='.png'):
        """
        Crée un DataFrame avec les chemins d'images et leurs labels (nom du sous-dossier).

        Args:
            data_path (str): Chemin vers le dossier contenant les sous-dossiers d'images.
            extensions (tuple): Extensions d'image valides à inclure.

        Returns:
            pd.DataFrame: DataFrame avec colonnes 'Image_Path' et 'Label'.
        """
        image_paths = []
        labels = []

        for label in os.listdir(data_path):
            # On parcours les fichier classés selon label
            label_path = os.path.join(data_path, label)
            if os.path.isdir(label_path):
                for filename in os.listdir(label_path):
                    # On récupères les images inclus dans le fichier
                    if filename.lower().endswith(extensions):
                        full_path = os.path.join(label_path, filename)
                        image_paths.append(full_path)
                        labels.append(label)

        return pd.DataFrame({
            "Image_Path": image_paths,
            "Label": labels
        })
    return (create_image_dataframe,)


@app.cell
def _(mo):
    mo.md(r"""# Parameters""")
    return


@app.cell
def _():
    # Data path parameters
    data_train_path = "Data/train"
    data_test_path = "Data/test"
    data_valid_path = "Data/valid"
    return data_test_path, data_train_path, data_valid_path


@app.cell
def _(mo):
    mo.md(r"""# Import the Data""")
    return


@app.cell
def _(
    create_image_dataframe,
    data_test_path,
    data_train_path,
    data_valid_path,
):
    data_train = create_image_dataframe(data_train_path)
    data_test = create_image_dataframe(data_test_path)
    data_valid = create_image_dataframe(data_valid_path)
    return data_test, data_train, data_valid


@app.cell
def _(data_train, display):
    display(data_train.head())
    return


@app.cell
def _(data_train):
    print("Nombre d'image train : ", len(data_train))
    print("Nombre d'image par label",data_train.Label.value_counts())

    return


@app.cell
def _(data_test):
    print("Nombre d'image test : ", len(data_test))
    print("Nombre d'image par label",data_test.Label.value_counts(normalize= True))
    return


@app.cell
def _(data_valid):
    print("Nombre d'image valid : ", len(data_valid))
    print("Nombre d'image par label",data_valid.Label.value_counts(normalize= True))

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
