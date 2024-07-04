# Data science - ASCAT slope/curvature #

## Summary ##

The package contains Python code to read ASCAT sigma0 time series data, compute and apply an azimuth angle correction, compute the ESD (Estimated Standard Deviation) and compute slope and curvature time series. The processing system is set up to process single grid points. The grid points can be selected using the [DGG Locator website](https://dgg.geo.tuwien.ac.at/). The ASCAT sigma0 data is based on the Fibonacci 6.25 km grid, which is important to be selected in the top left corner before starting searching for grid points. At the moment the ASCAT sigma0 data is limited to a small study area located in the US ().

## Install Python environment ##

Run the **makefile** in a terminal to install the Python environment called *my_env* in your home folder at *~/my_conda*:

> ```bash
> cd path/to/ds_slope_curvature
> make
> ```

Python dependencies required to run the example Jupyter notebook located at **src/test.ipynb** are defined in the **environment.yml** file. If any Python packages are missing for your analysis, you have the option to either manually install them using commands like `pip install` in the terminal (don't forget to activate the Python environment before), or add them to the **environment.yml** file.

## Remove Python environment ##

If you want to remove the Python environment you can use:

> ```bash
> make clean
> ```

## Running a Jupyter notebook using the correct kernel ##

The Python environment installation also add a kernel called **my_kernel**, that needs to be selected in the top right corner of a Jupyter notebook before executing a Jupyter notebook. The example notebook located at **src/test.ipynb** contains a very simple example how to call the processing steps in order to compute a slope and curvature climatology and time series.
