# Getting started #

## Data ##

The available datasets are: ASCAT, Era5, Era5-Land and GLDAS, each of them providing different data. Soil moisture data is available for each dataset, note that it is acquired differently depending on the dataset. You can find out more about each dataset in the "Metadata" folder.

Note that these Datasets don't have the same file-format as Sentinel-1 or -2 data: In contrast to .tif files where an image is stored in raster format, a netCDF4 file represents a 5x5 degree cell (https://ecmwf-models.readthedocs.io/en/latest/_images/5x5_cell_partitioning.png) containing multiple time series (grid points). Each time series is then stored as a 1D-array which is either sorted by grid point (contiguous ragged array) or not sorted at all (indexed ragged array) (https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html). 

## Set up ##

In your home directory you can see the folders "lectures" and "shared". Under "shared/datasets/fe/data" you can find sentinel-1 and -2 data. However in the following Notebooks we will handle ASCAT, Era5, Era5-Land and GLDAS data, you can find these datasets under "shared/dataset/rs". To work with these datasets, first create a new folder in your home directory, only you have access to this folder and you should write all your code there. To get started copy the provided notebooks to your personal folder, you can run and edit them there.

## Creating a new kernel ## 

To set up a new kernel, with all the modules needed already installed, navigate to the "environment" folder in terminal and run the command *make*. This might take a few minutes and will install a new kernel (datascience-env) which you can select in the top right corner of jupyter when running your notebooks. To remove the kernel once already installed, run *make clean*. If you can't find the new kernel try restarting the jupyter server.

## Launcher ##

When writing your own code you can either launch a Notebook directly in Jupyter, or use VS-Code. To use VS-Code launch a Code Server Notebook, a new tab will then open in your browser. Press *F1* and type and select *File: Open Folder* to navigate to your direcotry in your explorer. You can now create and edit existing files in VS-Code, before running files you need to select a kernel - to do this, press *select kernel* in the top right corner of an open notebook, then press *Python Environments* and finally select *datascience-env*. You can us VS-Code to write and run your code. Beware that you can't pip install packages in a Notebook with VS-Code, to install packages press *Ctrl+Shift+`* and write the pip install code in the terminal directly.

## Git ##

If you want to use git with JupyterLab or VS-Code you have to move to a terminal window.

1. Navigate to your your directory with *cd &lt;path to your directory&gt;*
2. Clone your existing repository with *git clone &lt;repository URL&gt;*
3. Initialize with *git init*
4. Define where you want to push your files *git remote add origin cd &lt;path to your directory&gt;*
5. Add files with *git add &lt;file_name&gt;*
6. Commit added files with *git commit -m "commit message"*
7. Push files to the repository with *git push &lt;repository URL&gt;*


Note:
- You can always check out the status of your files with *git status*
- The files will be pushed to a new branch named "master"

Other useful commands:
- *git init* to create a git repo
- *git add &lt;file_name&gt;* to put files in the stage'ing state (prepareing them for the commit)
- *git commit -m "commit message"* to commit all staged files with an apropriate commit message
- *git commit -a -m "commit message"* to add and commit all modified files
- *git clone &lt;repository URL&gt;* to clone a repository
- *git push* to save local changes to remote repository
- *git pull* to load newest version from remote repository
- *git status* to see the status of your files
- *git log* shows the previous commits
- *git checkout &lt;branch_name&gt;* changes to another branch
- *git checkout -b &lt;branch_name&gt;* git creates and changes to that branch
- *git branch &lt;branch_name&gt;* creates a new branch
- *git branch -d &lt;branch_name&gt;* deletes a branch
- *git merge &lt;branch_name&gt;* branch gets merged with main branch 
- *git rm &lt;file_name&gt;* to delete a file
- *git help -a* show a list of available commands