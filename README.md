# CV Skin_Cancer
## Deep Learning Project on Skin Cancer

More than 50% of lesions are confirmed through histopathology (histo), the ground truth for the rest of the cases is either follow-up examination (follow_up), expert consensus (consensus), or confirmation by in-vivo confocal microscopy (confocal).


Create a virtual environment with 
 "conda create -n cv_lung_project python=3.11"

 and then 

 `conda activate cv_lung_project python=3.11`


## Download the data
Make sure git-lfs is installed (https://git-lfs.com)
`git lfs install`

Then you can just downloa the data for the project with 
`git clone https://huggingface.co/datasets/marmal88/skin_cancer`

If you want to clone without large files - just their pointers
`GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/marmal88/skin_cancer`


## Notebook Usage
For notebook we used Marimo which is git friendly, if you want a Jupyter notebook : 
`marimo convert your_notebook.ipynb > your_notebook.py`

To run the notebook, 
`marimo run your_notebook.py`

To edit : 
`marimo edit`

https://github.com/marimo-team/marimo for more information 



