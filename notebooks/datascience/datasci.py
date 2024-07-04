import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from cadati.jd_date import jd2dt
from cadati.np_date import dt2jd
import matplotlib.pyplot as plt
from fibgrid.realization import FibGrid
from pynetcf.time_series import GriddedNcContiguousRaggedTs
from pynetcf.time_series import GriddedNcIndexedRaggedTs
from pynetcf.time_series import GriddedNcTs
from pynetcf.time_series import OrthoMultiTs
from pygeogrids.netcdf import load_grid

from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

def format_date(xticks, pos=None):
    """
    Define date format of time axis.

    Parameters
    ----------
    xticks : numpy.ndarray
        xticks
    """
    dt = mdates.num2date(xticks)

    if dt.month == 7:
        fmt = "%Y"
        return f"{dt.strftime(fmt)}"
    else:
        return ""


def set_xaxis_formatter(ax):
    """
    Format time series axis.

    Parameters
    ----------
    ax : matplotlib.axes
    """
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mticker.NullFormatter())
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_minor_formatter(mticker.FuncFormatter(format_date))


def days2dt(days, ref=np.datetime64("1900-01-01")):
    """
    Fraction of days to datetime.

    Parameters
    ----------
    days : numpy.ndarray
        Fraction of days since reference time stamp.
    ref : numpy.datetime64, optional
        Reference time stamp where days started counting.

    Returns
    -------
    dt : numpy.ndarray
        Datetime array.
    """
    return ref + (days * 24. * 3600 * 1e6).astype("timedelta64[us]")


class AscatData(GriddedNcContiguousRaggedTs):
    """
    Class reading ASCAT SSM 6.25 km data.
    """

    def __init__(self, path, read_bulk=True):
        """
        Initialize ASCAT data.

        Parameters
        ----------
        path : str
            Path to dataset.
        read_bulk : bool, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        """
        grid = FibGrid(6.25)
        ioclass_kws = dict(read_bulk=read_bulk)
        super().__init__(path, grid, ioclass_kws=ioclass_kws)


class AscatDataH121(GriddedNcIndexedRaggedTs):
    """
    Class reading ASCAT Data.
    """

    def __init__(self, path, read_bulk=False):
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


class AscatSig0Data(GriddedNcContiguousRaggedTs):
    """
    Class reading ASCAT sigma0 6.25 km data.
    """

    def __init__(self, path, read_bulk=True):
        """
        Initialize ASCAT data.

        Parameters
        ----------
        path : str
            Path to dataset.
        read_bulk : bool, optional
            If "True" all data will be read in memory, if "False"
            only a single time series is read (default: False).
            Use "True" to process multiple GPIs in a loop and "False" to
            read/analyze a single time series.
        """
        grid = FibGrid(6.25)
        ioclass_kws = dict(read_bulk=read_bulk)
        super().__init__(path, grid, ioclass_kws=ioclass_kws)

    def _read_gp(self, gpi, **kwargs):
        """
        Method reads data for given gpi, additional keyword arguments
        are passed to ioclass.read

        Parameters
        ----------
        gp : int
            Grid point.

        Returns
        -------
        ts : pandas.DataFrame
            Time series data.
        """
        if self.mode in ["w", "a"]:
            raise IOError("trying to read file is in write/append mode")

        if not self._open(gpi):
            return None

        if self.parameters is None:
            try:
                data = self.fid.read_all(gpi, dates_direct=True, **kwargs)
            except OSError:
                return None
        else:
            data = self.fid.read(self.parameters, gpi, **kwargs)

        return data

        fill_values = []
        dtypes = []

        predefined_fill_values = {
            "backscatter_for": float32_nan,
            "backscatter_mid": float32_nan,
            "backscatter_aft": float32_nan,
            "incidence_angle_for": float32_nan,
            "incidence_angle_mid": float32_nan,
            "incidence_angle_aft": float32_nan,
            "azimuth_angle_for": float32_nan,
            "azimuth_angle_mid": float32_nan,
            "azimuth_angle_aft": float32_nan,
            "kp_for": float32_nan,
            "kp_mid": float32_nan,
            "kp_aft": float32_nan,
            "time": 0,
            "gpi": int32_nan,
            "swath_indicator": uint8_nan,
            "as_des_pass": uint8_nan,
            "sat_id": uint8_nan
        }

        lut_rename = {
            "backscatter_for": "sigf",
            "backscatter_mid": "sigm",
            "backscatter_aft": "siga",
            "incidence_angle_for": "incf",
            "incidence_angle_mid": "incm",
            "incidence_angle_aft": "inca",
            "azimuth_angle_for": "azif",
            "azimuth_angle_mid": "azim",
            "azimuth_angle_aft": "azia",
            "kp_for": "kpf",
            "kp_mid": "kpm",
            "kp_aft": "kpa",
            "gpi": "gpi",
            "swath_indicator": "swath_indicator",
            "as_des_pass": "as_des_pass",
            "time": "time",
            "sat_id": "sat_id"
        }

        for name in data.keys():
            if name in predefined_fill_values.keys():
                if "time" in name:
                    fill_values.append(0)
                    dtypes.append((name, "<M8[us]"))
                else:
                    fill_values.append(predefined_fill_values[name])
                    dtypes.append((lut_rename[name], data[name].dtype.str))

        # compute julian date
        jd_nan = dt2jd(np.datetime64("1970-01-01"))
        fill_values.append(jd_nan)
        dtypes.append(("jd", "<f8"))

        # create new masked numpy array
        ts = np.ma.zeros(data["time"].size, dtype=np.dtype(dtypes))
        ts.set_fill_value(fill_values)
        ts[:] = np.ma.masked

        # copy data to new masked numpy array
        for name in data.keys():
            if name in predefined_fill_values.keys():
                if "time" in name:
                    ts[name] = days2dt(data[name])
                else:
                    ts[lut_rename[name]] = data[name]

        ts["jd"] = dt2jd(ts["time"])

        return ts


class Era5Land(GriddedNcTs):
    """
    Read time series data from ERA5 netCDF files.
    """

    def __init__(self, path, read_bulk=True, celsius=True):
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

    def __init__(self, path, read_bulk=True, celsius=True):
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

    def __init__(self, path, read_bulk=True, celsius=True):
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


def read_grid_point_example(loc,
                            ascat_sm_path,
                            era5_land_path=None,
                            era5_path=None,
                            gldas_path=None,
                            ascat_sigma0_path=None,
                            read_bulk=False):
    """
    Read grid point for given lon/lat coordinates or grid_point.

    Parameters
    ----------
    loc : int, tuple
        Tuple is interpreted as longitude, latitude coordinate.
        Integer is interpreted as grid point index.
    ascat_sm_path : str
        Path to ASCAT soil moisture data.
    era5_land_path : str
        Path to ERA5-Land data.
    era5_path : str
        Path to ERA5 data.
    gldas_path : str
        Path to GLDAS data.
    ascat_sigma0_path : str
        Path to ASCAT backscatter data.
    read_bulk : bool, optional
        If "True" all data will be read in memory, if "False"
        only a single time series is read (default: False).
        Use "True" to process multiple GPIs in a loop and "False" to
        read/analyze a single time series.
    """
    data = {}

    print(f"Reading ASCAT soil moisture: {ascat_sm_path}")
    ascat_obj = AscatDataH121(ascat_sm_path, read_bulk)

    if isinstance(loc, tuple):
        lon, lat = loc
        ascat_gpi, distance = ascat_obj.grid.find_nearest_gpi(lon, lat)
        print(f"ASCAT GPI: {ascat_gpi} - distance: {distance:8.3f} m")
    else:
        ascat_gpi = loc
        lon, lat = ascat_obj.grid.gpi2lonlat(ascat_gpi)
        print(f"ASCAT GPI: {ascat_gpi}")

    ascat_ts = ascat_obj.read(ascat_gpi)

    if ascat_ts is None:
        raise RuntimeError(f"ASCAT soil moisture data not found: {ascat_sm_path}")

    # set observations to NaN with less then two observations
    valid = ascat_ts["num_sigma"] >= 2
    ascat_ts.loc[~valid, ["sm", "sigma40", "slope40", "curvature40"]] = np.nan

    # read ASCAT backscatter data
    if ascat_sigma0_path is not None:
        print(f"Reading ASCAT backscatter: {ascat_sm_path}")
        ascat_sig0_obj = AscatSig0Data(ascat_sigma0_path, read_bulk)
        ascat_sig0_ts = ascat_sig0_obj.read(ascat_gpi)
        ascat_sig0_ts = pd.DataFrame(ascat_sig0_ts,
                                     index=days2dt(ascat_sig0_ts["time"]))

        sigma0_fields = [
            "backscatter_mid", "backscatter_aft", "backscatter_for"
        ]
        valid = np.sum(ascat_sig0_ts[sigma0_fields] > -30, axis=1) >= 2
        ascat_sig0_ts.loc[~valid, sigma0_fields] = np.nan

        inc_fields = [
            "incidence_angle_mid", "incidence_angle_aft", "incidence_angle_for"
        ]
        for inc_field in inc_fields:
            ascat_sig0_ts.loc[ascat_sig0_ts[inc_field] > 90,
                              inc_field] = np.nan

        ascat_ts.index = ascat_ts.index.astype("datetime64[ns]")
        ascat_sig0_ts.index = ascat_sig0_ts.index.astype("datetime64[ns]")

        # merge ascat soil moisture and backscatter
        ascat_ts = pd.merge_asof(ascat_ts,
                                 ascat_sig0_ts,
                                 left_index=True,
                                 right_index=True,
                                 suffixes=("", "_y"),
                                 tolerance=pd.Timedelta("2m"),
                                 direction="nearest")

    data["ascat_ts"] = ascat_ts
    data["ascat_gpi"] = ascat_gpi
    data["ascat_lon"] = lon
    data["ascat_lat"] = lat

    if era5_land_path is not None:
        print(f"Reading ERA5-Land: {era5_land_path}")
        era5_land_obj = Era5Land(era5_land_path, read_bulk)
        era5_land_gpi, distance = era5_land_obj.grid.find_nearest_gpi(lon, lat)
        era5_land_lon, era5_land_lat = era5_land_obj.grid.gpi2lonlat(
            era5_land_gpi)
        era5_land_ts = era5_land_obj.read(era5_land_gpi)
        print(f"ERA5-Land GPI: {era5_land_gpi} - distance: {distance:8.3f} m")
    else:
        era5_land_ts = None
        era5_land_gpi = None
        era5_land_lon = None
        era5_land_lat = None
        print(f"Warning: ERA5-Land not found: {era5_land_path}")

    data["era5_land_ts"] = era5_land_ts
    data["era5_land_gpi"] = era5_land_gpi
    data["era5_land_lon"] = era5_land_lon
    data["era5_land_lat"] = era5_land_lat

    if era5_path is not None:
        print(f"Reading ERA5: {era5_path}")
        era5_obj = Era5(era5_path, read_bulk)
        era5_gpi, distance = era5_obj.grid.find_nearest_gpi(lon, lat)
        era5_lon, era5_lat = era5_obj.grid.gpi2lonlat(era5_gpi)
        era5_ts = era5_obj.read(era5_gpi)
        print(f"ERA5 GPI: {era5_gpi} - distance: {distance:8.3f} m")
    else:
        era5_ts = None
        era5_gpi = None
        era5_lon = None
        era5_lat = None
        print(f"Warning: ERA5 not found: {era5_land_path}")

    data["era5_ts"] = era5_ts
    data["era5_gpi"] = era5_gpi
    data["era5_lon"] = era5_lon
    data["era5_lat"] = era5_lat

    if era5_ts is not None:

        ts = pd.merge_asof(data["ascat_ts"],
                           data["era5_ts"],
                           left_index=True,
                           right_index=True,
                           tolerance=pd.Timedelta("3h"),
                           direction="nearest")

        # mask data that is either frozen (temperature below 0) or with snow
        not_valid = (ts["stl1"] < 0) | (ts["sd"] > 0)
        data["ascat_ts"]["sm_valid"] = ~not_valid
    else:
        data["ascat_ts"]["sm_valid"] = True
        print("Warning: ERA5 not found - ASCAT soil moisture not masked!")

    if gldas_path is not None:
        print(f"Reading GLDAS: {gldas_path}")
        gldas_obj = Gldas(gldas_path, read_bulk)
        gldas_gpi, distance = gldas_obj.grid.find_nearest_gpi(lon, lat)
        gldas_lon, gldas_lat = gldas_obj.grid.gpi2lonlat(gldas_gpi)
        gldas_ts = gldas_obj.read(gldas_gpi)
        print(f"Noah GLDAS GPI: {gldas_gpi} - distance: {distance:8.3f} m")
    else:
        gldas_ts = None
        gldas_gpi = None
        gldas_lon = None
        gldas_lat = None
        print(f"Warning: GLDAS not found: {gldas_path}")

    data["gldas_ts"] = gldas_ts
    data["gldas_gpi"] = gldas_gpi
    data["gldas_lon"] = gldas_lon
    data["gldas_lat"] = gldas_lat

    return data


def plot_ascat_ts(ts, gpi, lon, lat, figsize=(15, 10)):
    """
    Plot ASCAT time series.

    Parameters
    ----------
    ts : pandas.DataFrame
        ASCAT time series data.
    gpi : int
        Grid point index.
    lon : float
        Longitude coordinate.
    lat : float
        Latitude coordinate.
    figsize : tuple of int, optional
        Figure size (default: (15, 10)).
    """
    print("Plot ASCAT time series...")
    fig, ax = plt.subplots(4, 1, figsize=figsize, sharex=True)

    data_info = "Metop ASCAT"
    location_info = f"GPI: {gpi} Lon: {lon:5.3f} Lat: {lat:5.3f}"

    ax[0].set_title(f"{data_info} Backscatter40 | {location_info}")
    ax[0].plot(ts["sigma40"], c="C4", lw=0.5, alpha=0.5)
    ax[0].plot(ts["sigma40"], c="C4", ls="none", marker=".")
    ax[0].set_ylabel("Backscatter (dB)")

    ax[1].set_title(f"{data_info}  Slope40 | {location_info}")
    ax[1].plot(ts["slope40"], c="C3")
    ax[1].set_ylabel("Slope (dB/deg)")

    ax[2].set_title(f"{data_info} Curvature40 | {location_info}")
    ax[2].plot(ts["curvature40"], c="C2")
    ax[2].set_ylabel("Curvature (dB/deg$^{2}$)")

    ax[2].set_title(f"{data_info} Soil Moisture | {location_info}")
    ax[3].plot(ts["sm"], c="C1", lw=0.5, alpha=0.5)
    ax[3].plot(ts["sm"], c="C1", ls="none", marker=".")
    ax[3].set_ylabel("Saturation (%)")

    set_xaxis_formatter(ax[0])
    for axes in ax:
        axes.grid(True)

    plt.tight_layout()


def plot_all_ts(data, figsize=(15, 10)):
    """
    Plot all time series data.

    Parameters
    ----------
    data : pandas.DataFrame
        Time series data.
    figsize : tuple of int, optional
        Figure size (default: (15, 10)).
    """
    print("Plot all time series...")
    fig, ax = plt.subplots(6, 1, figsize=figsize, sharex=True)

    title = "Metop ASCAT - Surface Soil Moisture"
    loc_info = f"GPI: {data['ascat_gpi']} Lon: {data['ascat_lon']:5.3f} Lat: {data['ascat_lat']:5.3f}"
    ax[0].set_title(f"{title} | {loc_info}")
    ax[0].plot(data["ascat_ts"]["sm"], c="C1", lw=0.5, alpha=0.7)
    ax[0].plot(data["ascat_ts"]["sm"][data["ascat_ts"]["sm_valid"]],
               c="C1",
               ls="none",
               marker=".",
               label="ASCAT SM valid")
    ax[0].plot(data["ascat_ts"]["sm"][~data["ascat_ts"]["sm_valid"]],
               c="C3",
               ls="none",
               marker=".",
               label="ASCAT SM not valid")
    ax[0].set_ylabel("Saturation (%)")
    ax[0].legend()

    title = "ERA5 - Soil temperature Level 1 (stl1)"
    loc_info = f"GPI: {data['era5_gpi']} Lon: {data['era5_lon']:5.3f} Lat: {data['era5_lat']:5.3f}"
    ax[1].set_title(f"{title} | {loc_info}")
    ax[1].plot(data["era5_ts"]["stl1"], color="C5")
    ax[1].set_ylabel("degree Celsius")

    title = "ERA5 - Volumetric soil water layer 1 (swvl1)"
    loc_info = f"GPI: {data['era5_gpi']} Lon: {data['era5_lon']:5.3f} Lat: {data['era5_lat']:5.3f}"
    ax[2].set_title(f"{title} | {loc_info}")
    ax[2].plot(data["era5_ts"]["swvl1"])
    ax[2].set_ylabel("Vol. SM (m$^{3}$ m$^{-3}$)")

    title = "ERA5-Land - Volumetric soil water layer 1 (swvl1)"
    loc_info = f"GPI: {data['era5_land_gpi']} Lon: {data['era5_land_lon']:5.3f} Lat: {data['era5_land_lat']:5.3f}"
    ax[3].set_title(f"{title} | {loc_info}")
    ax[3].plot(data["era5_land_ts"]["swvl1"])
    ax[3].set_ylabel("Vol. SM (m$^{3}$ m$^{-3}$)")

    title = "ERA5-Land - Total precipitation (tp)"
    loc_info = f"GPI: {data['era5_land_gpi']} Lon: {data['era5_land_lon']:5.3f} Lat: {data['era5_land_lat']:5.3f}"
    ax[4].set_title(f"{title} | {loc_info}")
    ax[4].plot(data["era5_land_ts"]["tp"])
    ax[4].set_ylabel("Accumulated rain (m)")

    title = "GLDAS Noah Land Surface Model V2.1 - Soil moisture 0-10 cm (SoilMoi0_10cm_inst)"
    loc_info = f"GPI: {data['gldas_gpi']} Lon: {data['gldas_lon']:5.3f} Lat: {data['gldas_lat']:5.3f}"
    ax[5].set_title(f"{title} | {loc_info}")
    ax[5].plot(data["gldas_ts"]["SoilMoi0_10cm_inst"], c="C3")
    ax[5].set_ylabel("SM (kg m$^{-2}$)")

    set_xaxis_formatter(ax[0])
    for axes in ax:
        axes.grid(True)

    plt.tight_layout()


def plot_scatter_plot(data, figsize=(10, 10)):
    """
    Plot ASCAT vs ERA5 scatterplot.

    Parameters
    ----------
    data : pandas.DataFrame
        Time series data.
    figsize : tuple of int, optional
        Figure size (default: (10, 10)).
    """
    print("Plot scatterplot...")

    data["era5_ts"].index = data["era5_ts"].index.astype("datetime64[ns]")

    ts = pd.merge_asof(data["ascat_ts"],
                       data["era5_ts"],
                       left_index=True,
                       right_index=True,
                       tolerance=pd.Timedelta("3h"),
                       direction="nearest")

    kwargs = {"facecolors": "None", "edgecolor": "C0"}
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data_info = "Metop ASCAT - Surface Soil Moisture"
    loc_info = f"GPI: {data['ascat_gpi']} Lon: {data['ascat_lon']:5.3f} Lat: {data['ascat_lat']:5.3f}"

    ax.set_title(f"{data_info} \n {loc_info}")
    ax.scatter(ts["sm"][ts["sm_valid"]], ts["swvl1"][ts["sm_valid"]], **kwargs)
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axline((xlim[0], ylim[0]), (xlim[1], ylim[1]),
              color="black",
              linestyle=(0, (5, 5)))
    ax.set_xlabel("ASCAT soil moisture")
    ax.set_ylabel("ERA5 swvl1")


def plot_ascat_histogram(ts, gpi, lon, lat, figsize=(10, 10)):
    """
    Plot ASCAT soil moisture histogram.

    Parameters
    ----------
    ts : pandas.DataFrame
        ASCAT time series data.
    gpi : int
        Grid point index.
    lon : float
        Longitude coordinate.
    lat : float
        Latitude coordinate.
    figsize : tuple of int, optional
        Figure size (default: (10, 10)).
    """
    print("Plot histogram...")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    data_info = "Metop ASCAT - Surface Soil Moisture"
    loc_info = f"GPI: {gpi} Lon: {lon:5.3f} Lat: {lat:5.3f}"

    bins = np.linspace(0, 100, 40)
    ax.set_title(f"{data_info} \n {loc_info}")
    ax.hist(ts["sm"][ts["sm_valid"]], bins=bins, density=True, alpha=0.65)
    ax.set_xlabel("ASCAT soil moisture")

    quantiles = [5, 25, 50, 75, 95]
    alpha_list = [0.6, 0.8, 1, 0.8, 0.6]
    ymax_list = [0.4, 0.5, 0.6, 0.7, 0.8]
    ylim = ax.get_ylim()

    for quant, alpha, ymax in zip(quantiles, alpha_list, ymax_list):
        quant_value = np.nanpercentile(ts["sm"].values, quant)
        ax.axvline(quant_value, alpha=alpha, ymax=ymax, linestyle=":")
        ax.text(quant_value,
                ymax * ylim[1] + ylim[1] * 0.05,
                f"{quant}th",
                size=10,
                va="center",
                ha="center")


def plot_sig_inc_interactive(data):
    """
    Plot interactive sigma0 vs incidence angle scatterplot using mouse motion
    on a time series.

    Parameters
    ----------
    data : pandas.DataFrame
        Time series data.
    """
    p1 = SigIncPlot(data["ascat_ts"], data["ascat_gpi"], data["ascat_lon"],
                    data["ascat_lat"])

    return p1


class SigIncPlot:
    """
    Create interactive plot showing ASCAT sigma40, slope, curvature and
    soil moisture time series, as well as backscatter vs incidence angle and
    local slopes.
    """

    def __init__(self, ts, gpi, lon, lat, figsize=(18, 10)):
        """
        Initialize data.

        Parameters
        ----------
        ts : pandas.DataFrame
            ASCAT time series data.
        gpi : int
            Grid point index.
        lon : float
            Longitude coordinate.
        lat : float
            Latitude coordinate.
        figsize : tuple of int, optional
            Figure size (default: (18, 10)).
        """
        self.ind_last = 0

        self.ts = ts.replace(np.iinfo(np.int32).min, np.nan)

        self.ref_angle = 40.
        self.inc_angle = np.arange(20, 65, 1)[:, np.newaxis]
        self.ts["num_date"] = mdates.date2num(ts.index.values.data)

        fields = ["backscatter_mid", "backscatter_for", "backscatter_aft"]
        y_min = self.ts[fields].min().min()
        y_max = self.ts[fields].max().max()

        self.sigma0_lim = [
            y_min - np.abs(y_min) * 0.05,
            y_max - np.abs(y_max) * 0.05,
        ]

        y_min = []
        y_max = []
        for i, beam in enumerate(["for", "aft"]):
            x, y = compute_local_slope(self.ts["backscatter_mid"],
                                       self.ts[f"backscatter_{beam}"],
                                       self.ts["incidence_angle_mid"],
                                       self.ts[f"incidence_angle_{beam}"])
            y_min.append(y.min())
            y_max.append(y.max())

        y_min = min(y_min)
        y_max = max(y_max)

        self.local_slopes_lim = [
            y_min - np.abs(y_min) * 0.05,
            y_max - np.abs(y_max) * 0.05,
        ]

        self.fig = plt.figure(figsize=figsize)

        gs = GridSpec(4, 8, figure=self.fig)
        ax1 = self.fig.add_subplot(gs[0, :5])
        ax2 = self.fig.add_subplot(gs[1, :5], sharex=ax1)
        ax3 = self.fig.add_subplot(gs[2, :5], sharex=ax2)
        ax4 = self.fig.add_subplot(gs[3, :5], sharex=ax3)
        ax5 = self.fig.add_subplot(gs[:2, 5:])
        ax6 = self.fig.add_subplot(gs[2:, 5:], sharex=ax5)
        self.ax = [ax1, ax2, ax3, ax4, ax5, ax6]

        title = "Metop ASCAT - Backscatter40"
        loc_info = f"GPI: {gpi} Lon: {lon:5.3f} Lat: {lat:5.3f}"
        self.ax[0].set_title(f"{title} | {loc_info}")

        self.ax[0].plot(self.ts["sigma40"], c="C4", lw=0.5, alpha=0.5)
        self.ax[0].plot(self.ts["sigma40"], c="C4", ls="none", marker=".")
        self.ax[0].set_ylabel("Backscatter (dB)")

        title = "Metop ASCAT - Slope"
        self.ax[1].set_title(f"{title} | {loc_info}")
        self.ax[1].plot(self.ts["slope40"], c="C3")
        self.ax[1].set_ylabel("Slope (dB/deg)")

        title = "Metop ASCAT - Curvature"
        self.ax[2].set_title(f"{title} | {loc_info}")
        self.ax[2].plot(self.ts["curvature40"], c="C2")
        self.ax[2].set_ylabel("Curvature (dB/deg$^{2}$)")

        title = "Metop ASCAT - Surface Soil Moisture"
        self.ax[3].set_title(f"{title} | {loc_info}")
        self.ax[3].plot(self.ts["sm"], c="C1", lw=0.5, alpha=0.5)
        self.ax[3].plot(self.ts["sm"], c="C1", ls="none", marker=".")
        self.ax[3].set_ylabel("Saturation (%)")

        set_xaxis_formatter(self.ax[0])
        for axes in self.ax:
            axes.grid(True)

        self.cid = self.fig.canvas.mpl_connect('motion_notify_event',
                                               self.on_move)

        plt.subplots_adjust(top=0.965,
                            bottom=0.039,
                            left=0.057,
                            right=0.986,
                            hspace=0.301,
                            wspace=0.6)

        self.line = {}

    def on_move(self, event):
        """
        on_move mouse event.

        Parameters
        ----------
        event
        """
        if event.inaxes in self.ax and event.inaxes not in [
                self.ax[4], self.ax[5]
        ]:
            x = event.xdata
            distances = np.abs(x - self.ts["num_date"])
            self.ind_last = distances.argmin()
            self.update()

    def update(self):
        """
        Update sigma0 vs incidence angle.
        """
        delta_angle = self.inc_angle - self.ref_angle
        data = self.ts.iloc[self.ind_last]

        for k, line in self.line.items():
            line.remove()

        for i, ax in enumerate(
            [self.ax[0], self.ax[1], self.ax[2], self.ax[3]]):
            self.line[i] = ax.axvline(days2dt(data["time"]))

        window_subset = np.abs(self.ts.index -
                               self.ts.index[self.ind_last]) < np.timedelta64(
                                   21, "D")
        data_window = self.ts[window_subset]

        sig0 = data["sigma40"] + data["slope40"] * delta_angle + 0.5 * data[
            "curvature40"] * delta_angle**2

        self.ax[4].clear()
        self.ax[4].plot(self.inc_angle,
                        sig0.flatten(),
                        lw=2,
                        color="C3",
                        label="sigma0 vs incidence angle model")

        subset = (
            (data_window["backscatter_mid"] < data_window["backscatter_for"]) |
            (data_window["backscatter_mid"] < data_window["backscatter_aft"]))

        kwargs = {"ls": "none", "marker": "o"}
        colors = ["C0", "C1", "C2"]
        for i, beam in enumerate(["for", "mid", "aft"]):
            self.ax[4].plot(
                data_window[f"incidence_angle_{beam}"][subset],
                data_window[f"backscatter_{beam}"][subset],
                color=colors[i],  #label=f"sigma0 {beam}",
                **kwargs)

        kwargs = {"ls": "none", "fillstyle": "none", "marker": "o"}
        for i, beam in enumerate(["for", "mid", "aft"]):
            self.ax[4].plot(data_window[f"incidence_angle_{beam}"][~subset],
                            data_window[f"backscatter_{beam}"][~subset],
                            color=colors[i],
                            label=f"sigma0 {beam}",
                            **kwargs)

        self.ax[4].set_ylim(self.sigma0_lim)
        self.ax[4].set_xlim(15, 70)

        y = data_window[[
            "backscatter_for", "backscatter_mid", "backscatter_aft"
        ]].values.flatten()
        x = data_window[[
            "incidence_angle_for", "incidence_angle_mid", "incidence_angle_aft"
        ]].values.flatten() - self.ref_angle

        valid = ~np.isnan(x) & ~np.isnan(y)
        coeff = np.polyfit(x[valid], y[valid], 2)
        y_hat = coeff[0] * (self.inc_angle - 40)**2 + coeff[1] * (
            self.inc_angle - 40) + coeff[2]

        self.ax[4].plot(self.inc_angle,
                        y_hat,
                        lw=2,
                        color="C4",
                        label="2nd order fit")

        self.ax[5].clear()

        for i, beam in enumerate(["for", "aft"]):
            x, y = compute_local_slope(data_window["backscatter_mid"],
                                       data_window[f"backscatter_{beam}"],
                                       data_window["incidence_angle_mid"],
                                       data_window[f"incidence_angle_{beam}"])

            subset = y < 0

            kwargs = {"ls": "none", "fillstyle": "none", "marker": "o"}
            self.ax[5].plot(x[subset],
                            y[subset],
                            color=colors[i],
                            label=f"local slope mid-{beam} (<0)",
                            **kwargs)

            kwargs = {"ls": "none", "marker": "o"}
            self.ax[5].plot(x[~subset],
                            y[~subset],
                            color=colors[i],
                            label=f"local slope mid-{beam} (>0)",
                            **kwargs)

        y = data["slope40"] + data["curvature40"] * (self.inc_angle -
                                                     self.ref_angle)
        self.ax[5].plot(self.inc_angle,
                        y,
                        color="C3",
                        lw=2,
                        label="local slope regression fit")

        self.ax[5].set_ylim(self.local_slopes_lim)

        self.ax[4].set_title("Backscatter vs incidence angle")
        self.ax[4].set_ylabel("Backscatter (dB)")

        self.ax[5].set_title("Local slopes")
        self.ax[5].set_ylabel("Slope (dB/deg)")

        self.ax[4].legend()
        self.ax[5].legend()

        self.ax[4].grid(True)
        self.ax[5].grid(True)

        self.fig.canvas.draw()


def compute_local_slope(sig_mid, sig_x, inc_mid, inc_x):
    """
    Compute local slope.

    Parameters
    ----------
    sig_mid : numpy.ndarray
        Mid beam backscatter observations.
    sig_x : numpy.ndarray
        Fore/Aft beam backscatter observations.
    inc_mid : numpy.ndarray
        Mid beam incidence angle.
    inc_x : numpy.ndarray
        Fore/Aft beam incidence angle.

    Returns
    -------
    x : numpy.ndarray
        Local slope incidence angle.
    y : numpy.ndarray
        Local slope.
    """
    sig_diff = sig_mid - sig_x
    inc_diff = inc_mid - inc_x
    y = sig_diff / inc_diff
    x = (inc_mid + inc_x) / 2.

    return x, y


def set_paths():
    """
    Set paths to datasets.
    """
    # linux path example
    ascat_sig0_path = Path("/data-write/RADAR/hsaf/stack_cell_merged_metop_abc")
    ascat_sm_path = Path("/data-write/RADAR/warp/freaky_forge_2023/r18abc/081_ssm_userformat/datasets")
    ascat_sig0_path = Path("/home/shahn/media/ftp/data_science_down/datasets_v2/ascat/ascat_sigma0")
    ascat_sm_path = Path("/home/shahn/media/ftp/data_science_down/datasets_v2/ascat/ascat_freaky_forge_2023_r15abc")

    # root_path = Path("/data-read/RADAR/warp")
    root_path = Path("/home/shahn/media/ftp/data_science_down/datasets_v2")
    era5_land_path = root_path / "era5_land_2023"
    era5_path = root_path / "era5_2023"
    gldas_path = root_path / "gldas_2023"

    # windows path example
    # ascat_sig0_path = Path(r'\\project14\data-write\RADAR\hsaf\stack_cell_merged_metop_abc')
    # ascat_sm_path = Path(r"\\project14\data-write\RADAR\warp\freaky_forge_2023\r18abc\081_ssm_userformat\datasets")

    # root_path = Path(r"\\project14\data-read\RADAR\warp")
    # era5_land_path = root_path / "era5_land_2023"
    # era5_path = root_path / "era5_2023"
    # gldas_path = root_path / "gldas_2023"

    return ascat_sm_path, era5_land_path, era5_path, gldas_path, ascat_sig0_path


def main():
    """
    Main routine.
    """
    paths = set_paths()

    # https://dgg.geo.tuwien.ac.at/?grid=fibgrid_n6600000&gpi=2718380
    lat = 24.47
    lon = 84.36

    lat = 22.94
    lon = 89.94

    # wetland
    lat = 23.240
    lon = 89.977

    # no wetland
    # lat = 24.492
    # lon = 79.563

    # lat, lon =  22.739, 85.959
    # lon = 81.633
    # lat = 21.925
    loc = (lon, lat)

    # loc = 2450597
    # loc = 11010683

    plt.rcParams["backend"] = "tkagg"

    data = read_grid_point_example(loc, *paths)

    plot_ascat_ts(data["ascat_ts"], data["ascat_gpi"], data["ascat_lon"],
                  data["ascat_lat"])

    plot_all_ts(data)

    plot_ascat_histogram(data["ascat_ts"], data["ascat_gpi"],
                         data["ascat_lon"], data["ascat_lat"])

    plot_scatter_plot(data)

    p1 = SigIncPlot(data["ascat_ts"], data["ascat_gpi"], data["ascat_lon"],
                    data["ascat_lat"])

    plt.show()