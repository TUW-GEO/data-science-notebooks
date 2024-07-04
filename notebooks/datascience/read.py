import numpy as np
import pandas as pd
from datetime import datetime
import yaml
from pathlib import Path
import os

from cadati.jd_date import jd2dt
from cadati.np_date import dt2jd
from pynetcf.time_series import GriddedNcTs, GriddedNcIndexedRaggedTs, OrthoMultiTs,  GriddedNcContiguousRaggedTs
from pygeogrids.netcdf import load_grid
from fibgrid.realization import FibGrid

# Get the paths for each dataset:
source_path = Path('paths.yml').resolve()
with open(source_path, "r") as f:
    paths = yaml.safe_load(f)

for key in paths.keys():
    paths[key] = os.path.expanduser(paths[key])

class AscatDataDynSlope(GriddedNcContiguousRaggedTs):
    """
    Class reading ASCAT Data.
    """

    def __init__(self, path = paths["ascat_dyn_slope"], read_bulk=False):
        """
        Initialize ASCAT data.
        """
        grid = FibGrid(6.25)
        ioclass_kws = dict(read_bulk=read_bulk, obs_dim_name="obs")
        super().__init__(path, grid, ioclass_kws=ioclass_kws)

    def read(self, *args, **kwargs):
        ts = super().read(*args, **kwargs)
        if ts is not None:
            ts = ts.sort_index()

        return ts
        
class AscatDataH121(GriddedNcIndexedRaggedTs):
    """
    Class reading ASCAT Data.
    """

    def __init__(self, path = paths["ascat"], read_bulk=False):
        """
        Initialize ASCAT data.
        """
        grid = FibGrid(12.5)
        ioclass_kws = dict(read_bulk=read_bulk, obs_dim_name="obs")
        super().__init__(path, grid, ioclass_kws=ioclass_kws)

    def read(self, *args, **kwargs):
        ts = super().read(*args, **kwargs)
        if ts is not None:
            ts = ts.sort_index()

        return ts

class Era5Land(GriddedNcTs):
    """
    Read time series data from ERA5 netCDF files.
    """

    def __init__(self, path = paths["era5_land"], read_bulk=False, celsius=True):
        """
        Parameters
        ----------
        path : str
            Path to the data.
        read_bulk : boolean, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        celsius: boolean, optional
            if True temperature values are returned in degrees Celsius,
            otherwise they are in degrees Kelvin
            Default : True
        """
        parameter = ["t2m", "swvl1", "stl1", "tp"]

        grid_filename = os.path.join(path, "grid.nc")
        grid = load_grid(grid_filename)

        if type(parameter) != list:
            parameter = [parameter]

        self.parameters = parameter

        offsets = {}
        self.path = {}

        self.path['ts'] = path

        param_list = [
            '139', '167', '170', '183', '235', '236', 'stl1', '2t', 't2m'
        ]

        for parameter in self.parameters:
            if celsius and parameter in param_list:
                offsets[parameter] = -273.15
            else:
                offsets[parameter] = 0.0

        super(Era5Land, self).__init__(self.path['ts'],
                                       ioclass=OrthoMultiTs,
                                       grid=grid,
                                       ioclass_kws={'read_bulk': read_bulk},
                                       parameters=self.parameters,
                                       offsets=offsets)

    def read(self, *args, **kwargs):
        """
        Read method.

        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lon, lat coordinates and then reading.
        """
        if 'dates_direct' in kwargs.keys():
            ts = super(Era5Land, self).read(*args, **kwargs)
        else:
            ts = super(Era5Land, self).read(*args, dates_direct=True, **kwargs)
            ref_dt = np.datetime64(
                datetime.strptime(
                    self.fid.dataset.variables['time'].units[11:],
                    '%Y-%m-%d %H:%M:%S').isoformat())
            ts.index = jd2dt(ts.index.values + dt2jd(ref_dt))

        return ts

class Era5(GriddedNcTs):
    """
    Read time series data from ERA5 netCDF files,
    """

    def __init__(self, path = paths["era5"], read_bulk=False, celsius=True):
        """
        Parameters
        ----------
        path : str
            Path to dataset.
        read_bulk : bool, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        celsius: boolean, optional
            if True temperature values are returned in degrees Celsius,
            otherwise they are in degrees Kelvin
            Default : True
        """
        parameter = ["sd", "swvl1", "stl1", "t2m", "tp"]

        grid_filename = os.path.join(path, "grid.nc")
        grid = load_grid(grid_filename)

        if type(parameter) != list:
            parameter = [parameter]

        self.parameters = parameter

        offsets = {}
        self.path = {}

        self.path['ts'] = path

        param_list = [
            '139', '167', '170', '183', '235', '236', 'stl1', '2t', 't2m'
        ]

        for parameter in self.parameters:
            if celsius and parameter in param_list:
                offsets[parameter] = -273.15
            else:
                offsets[parameter] = 0.0

        super(Era5, self).__init__(self.path['ts'],
                                   ioclass=OrthoMultiTs,
                                   grid=grid,
                                   ioclass_kws={'read_bulk': read_bulk},
                                   parameters=self.parameters,
                                   offsets=offsets)

    def read(self, *args, **kwargs):
        """
        Read method.

        Takes either 1 or 2 arguments and calls the correct function
        which is either reading the gpi directly or finding
        the nearest gpi from given lon, lat coordinates and then reading.
        """
        if 'dates_direct' in kwargs.keys():
            ts = super(Era5, self).read(*args, **kwargs)
        else:
            ts = super(Era5, self).read(*args, dates_direct=True, **kwargs)
            ref_dt = np.datetime64(
                datetime.strptime(
                    self.fid.dataset.variables['time'].units[11:],
                    '%Y-%m-%d %H:%M:%S').isoformat())
            ts.index = jd2dt(ts.index.values + dt2jd(ref_dt))

        return ts

class Gldas(GriddedNcTs):
    """
    Read time series data from GLDAS netCDF files.
    """

    def __init__(self, path = paths["gldas"], read_bulk=True, celsius=True):
        """
        Initialize.

        Parameters
        ----------
        path : str
            Path to the data.
        read_bulk : boolean, optional
            if True the whole cell is read at once which makes bulk processing
            tasks that need all the time series of a cell faster.
            Default: False
        celsius: boolean, optional
            if True temperature values are returned in degrees Celsius,
            otherwise they are in degrees Kelvin. Default: True
        """
        parameter = ["SoilMoi0_10cm_inst", "Tair_f_inst"]

        grid_filename = os.path.join(path, "grid.nc")
        grid = load_grid(grid_filename)

        if type(parameter) != list:
            parameter = [parameter]

        self.parameters = parameter
        offsets = {}
        self.path = {'ts': path}

        for parameter in self.parameters:
            if celsius and parameter.startswith('SoilTMP') or \
               parameter.startswith('AvgSurfT') or \
               parameter.startswith('Tair'):
                offsets[parameter] = -273.15
            else:
                offsets[parameter] = 0.0

        super(Gldas, self).__init__(self.path['ts'],
                                    ioclass=OrthoMultiTs,
                                    grid=grid,
                                    ioclass_kws={'read_bulk': read_bulk},
                                    parameters=self.parameters,
                                    offsets=offsets)

def read_multiple_ds(loc,
                     ascat=None,
                     era5=None,
                     era5land=None,
                     gldas=None,
                     ref_ds=None):

    """
    Read and merge multiple time series data.

    Parameters
    ----------
    loc : int, tuple
        Tuple is interpreted as longitude, latitude coordinate.
        Integer is interpreted as grid point index.
    ascat : ascat object
    era5 : era5 object
    era5land : era5land object
    gldas : gldas object
    ref_ds: str
        Either "ascat", "era5", "era5land" or "gldas"
        Dataset from which gpi is chosen, and reference for the time index in the merged timeseries

    Returns
    ----------
    ts : pd.Dataframe
        Merged timeseries
    """

    # Throw Error if wrong ref_ds is chosen
    if ref_ds != "ascat" and ref_ds != "era5" and ref_ds != "era5land" and ref_ds != "gldas":
        raise SyntaxError("Choose either ascat, era5, era5land or gldas as a reference dataset")

    # Throw Error if no dataset is given
    if ascat is None and era5land is None and era5 is None and gldas is None:
        raise SyntaxError("No data")

    # Define initial empty Dataframes
    ascat_data, era5_data, era5land_data, gdal_data = None, None, None, None

    # Get the lon, lat coordinates out of loc, or from gpi depending on the ref_ds
    if isinstance(loc, tuple):
        lon, lat = loc
    elif ref_ds=="ascat":
        lon, lat = ascat.grid.gpi2lonlat(loc)
    elif ref_ds=="era5":
        lon, lat = era5.grid.gpi2lonlat(loc)
    elif ref_ds=="era5land":
        lon, lat = era5land.grid.gpi2lonlat(loc)
    elif ref_ds=="gldas":
        lon, lat = gldas.grid.gpi2lonlat(loc)

    ### Per Dataset:
    ### Find the nearest gpi to lon, lat; read the data and uniformally change the index to datetime64[ms] format
    
    if ascat:
        ascat_gpi, distance = ascat.grid.find_nearest_gpi(lon, lat)
        print(f"ASCAT GPI: {ascat_gpi} - distance: {distance:8.3f} m")
        ascat_data = ascat.read(ascat_gpi)
        ascat_data.index = ascat_data.index.astype('datetime64[ms]')
            
    if era5:
        era5_gpi, distance = era5.grid.find_nearest_gpi(lon, lat)
        print(f"ERA5 GPI: {era5_gpi} - distance: {distance:8.3f} m")
        era5_data = era5.read(era5_gpi)
        era5_data.index = era5_data.index.astype('datetime64[ms]')
        era5_data = era5_data.rename(columns={"swvl1":"swvl1_era5", "stl1":"stl1_era5", "t2m":"t2m_era5", "tp":"tp_era5"})

    if era5land:
        era5land_gpi, distance = era5land.grid.find_nearest_gpi(lon, lat)
        print(f"ERA5Land GPI: {era5land_gpi} - distance: {distance:8.3f} m")
        era5land_data = era5land.read(era5land_gpi)
        era5_data.index = era5_data.index.astype('datetime64[ms]')
        era5land_data = era5land_data.rename(columns={"t2m":"t2m_era5land", "swvl1":"swvl1_era5land", "stl1":"stl1_era5land", "tp":"tp_era5land"})
    
    if gldas:
        gldas_gpi, distance = gldas.grid.find_nearest_gpi(lon, lat)
        print(f"GLDAS GPI: {gldas_gpi} - distance: {distance:8.3f} m")
        gldas_data = gldas.read(gldas_gpi)
        gldas_data.index = gldas_data.index.astype('datetime64[ms]')

    ### Depending on ref_ds:
    ### Merge the data depending on which data is given
    
    if ref_ds=="ascat":
        ts=ascat_data
        if era5:
            ts = pd.merge_asof(ts,
                       era5_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if era5land:
            ts = pd.merge_asof(ts,
                       era5land_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if gldas:
            ts = pd.merge_asof(ts,
                       gldas_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")

    if ref_ds=="era5":
        ts=era5_data
        if ascat:
            ts = pd.merge_asof(ts,
                       ascat_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if era5land:
            ts = pd.merge_asof(ts,
                       era5land_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if gldas:
            ts = pd.merge_asof(ts,
                       gldas_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")

    if ref_ds=="era5land":
        ts=era5land_data
        if era5:
            ts = pd.merge_asof(ts,
                       era5_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if ascat:
            ts = pd.merge_asof(ts,
                       ascat,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if gldas:
            ts = pd.merge_asof(ts,
                       gldas_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")

    if ref_ds=="gldas":
        ts=gldas_data
        if era5:
            ts = pd.merge_asof(ts,
                       era5_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if era5land:
            ts = pd.merge_asof(ts,
                       era5land_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")
        if ascat:
            ts = pd.merge_asof(ts,
                       ascat_data,
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")

    return ts



































def read_multiple_datasets_close_to_eachother(loc,
                           ascat = None,
                           era5land=None,
                           era5=None,
                           gldas=None,
                           gpi_from = "ascat"):

    if ascat is None and era5_land is None and era5 is None and gldas is None:
        raise SyntaxError("No data")

    if gpi_from == "ascat":
        if ascat is None:
            raise SyntaxError("No ASCAT-Data")
        if isinstance(loc, tuple):
            lon, lat = loc
            ascat_gpi, distance = ascat.grid.find_nearest_gpi(lon, lat)
        else:
            ascat_gpi = loc
            lon, lat = ascat.grid.gpi2lonlat(ascat_gpi)
            
    elif gpi_from == "era5":
        if era5 is None:
            raise SyntaxError("No ERA5-Data")
        if isinstance(loc, tuple):
            lon, lat = loc
            era5_gpi, distance = era5.grid.find_nearest_gpi(lon, lat)
        else:
            era5_gpi = loc
            lon, lat = era5.grid.gpi2lonlat(era5_gpi)

    elif gpi_from == "era5land":
        if era5land is None:
            raise SyntaxError("No ERA5Land-Data")
        if isinstance(loc, tuple):
            lon, lat = loc
            era5land_gpi, distance = era5land.grid.find_nearest_gpi(lon, lat)
        else:
            era5land_gpi = loc
            lon, lat = era5land.grid.gpi2lonlat(era5land_gpi)

    elif gpi_from == "gldas":
        if gldas is None:
            raise SyntaxError("No GLDAS-Data")
        if isinstance(loc, tuple):
            lon, lat = loc
            gldas_gpi, distance = gldas.grid.find_nearest_gpi(lon, lat)
        else:
            gldas_gpi = loc
            lon, lat = era5land.grid.gpi2lonlat(gldas_gpi)

    if gpi_from == "ascat":
        lon_gpi, lat_gpi = ascat.grid.gpi2lonlat(ascat_gpi)
        if era5:
            era5_gpi =  era5.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if era5land:
            era5land_gpi = era5land.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if gldas:
            gldas_gpi = gldas.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]

    if gpi_from == "era5":
        lon_gpi, lat_gpi = era5.grid.gpi2lonlat(era5_gpi)
        if ascat:
            ascat_gpi =  ascat.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if era5land:
            era5land_gpi = era5land.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if gldas:
            gldas_gpi = gldas.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]

    if gpi_from == "era5land":
        lon_gpi, lat_gpi = era5land.grid.gpi2lonlat(era5land_gpi)
        if era5:
            era5_gpi =  era5.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if era5land:
            ascat_gpi = ascat.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if gldas:
            gldas_gpi = gldas.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]

    if gpi_from == "gldas":
        lon_gpi, lat_gpi = gldas.grid.gpi2lonlat(gldas_gpi)
        if era5:
            era5_gpi =  era5.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if era5land:
            era5land_gpi = era5land.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
        if ascat:
            ascat_gpi = ascat.grid.find_nearest_gpi(lon_gpi, lat_gpi)[0]
    
    print(f"ASCAT GPI: {ascat_gpi if ascat else None}\nERA5 GPI: {era5_gpi if era5 else None}\nERA5-Land GPI: {era5land_gpi if era5land else None}\nGLDAS GPI: {gldas_gpi if gldas else None}")























    