SHELL := /bin/bash
.DEFAULT_GOAL := all
environment_file := ~/shared/DataScience-RSP/OS/environment/environment.yml
local_conda_folder := ~/my_conda
env_name := datascience
kernel_name := datascience
.PHONY: setup clean
all: setup
setup:
	conda info -a
	conda install mamba -n base -c conda-forge
	mamba env create -f $(environment_file) -p $(local_conda_folder)
	$(local_conda_folder)/bin/python -m ipykernel install --user --name $(kernel_name) --display-name $(kernel_name)
clean:
	conda remove -p $(local_conda_folder) --all --yes
