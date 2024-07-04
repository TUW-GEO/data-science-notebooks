from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import mplcursors
from math import radians, sin, cos, sqrt, atan2
import numpy as np
from eomaps import Maps

def plot_gpis(loc, obj, k=100):
    """
    Plot the gpis and calculate the distance around a defined location.
    ----------
    Parameters:
    loc : tuple
        Longitude and latitude of a location
    obj : ASCAT, Era5, Era5-Land or GLDAS Object
    k : int, default: 100
        Number of grid points to be plotted
    """

    # Get index, distance to location, coordinates and cell number of closest gridpoints
    index, dist = obj.grid.find_k_nearest_gpi(loc[0], loc[1], k=k)
    lons, lats = obj.grid.gpi2lonlat(index)
    cells = obj.grid.gpi2cell(index)
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create a GeoAxes with a PlateCarree projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)    
    ax.add_feature(cfeature.STATES, alpha=0.2)

    # Set x and y-ticks and extent (+-5° from location)
    ax.set_xticks(range(-180, 181, 1), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 1), crs=ccrs.PlateCarree()) 
    ax.set_extent([loc[0]-5, loc[0]+5,loc[1]-5, loc[1]+5], crs=ccrs.PlateCarree())

    # Plot the gpis and location
    scatter = ax.scatter(lons, lats, marker = "x", color = "b", alpha = 0.4, zorder=998)
    ax.scatter(loc[0], loc[1], color="r", zorder=999)

    # Highlight closest gpi to location
    gpi = obj.grid.find_nearest_gpi(loc[0], loc[1])
    gpi_lon, gpi_lat = obj.grid.gpi2lonlat(gpi)
    ax.scatter(gpi_lon, gpi_lat, marker = "x", color = "b", alpha = 1, zorder=999)

    # Plot Gridlines of Cell
    ax.vlines(range(-180, 181, 5), -90,90, color = "r")
    ax.hlines(range(-90, 91, 5), -180,180, color = "r")

    # Add hovering feature showing gpi and distance
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"gpi: {index[0][sel.target.index]}\ncell: {cells[0][sel.target.index]}\ndistance: {round(dist[0][sel.target.index]/1000, 2)}km"))
    
    # Add a title and labels
    plt.title('Grid point Indexes')
    plt.xlabel("Longitude [°]")
    plt.ylabel("Latitude [°]")
    
    # Show the plot
    plt.show()

def plot_multiple_gpis(loc, obj1, obj2=None, obj3=None, obj4=None, k=30):
    """
    Plot the gpis in a radius of 0.5° around a defined location. And calculate the distance between each gpi and the location.
    ----------
    Parameters:
    loc : tuple containing lat and lon of a defined location
    obj : era5, era5-Land or ASCAT-Object
    """
    
    # Create a new figure
    plt.figure(figsize=(10, 6))
    
    # Create a GeoAxes with a PlateCarree projection
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.RIVERS)
    ax.add_feature(cfeature.LAKES)    
    ax.add_feature(cfeature.STATES, alpha=0.2)

    # Set x and y-ticks and extent (+-5° from location)
    ax.set_xticks(range(-180, 181, 1), crs=ccrs.PlateCarree())
    ax.set_yticks(range(-90, 91, 1), crs=ccrs.PlateCarree()) 
    ax.set_extent([loc[0]-1, loc[0]+1,loc[1]-1, loc[1]+1], crs=ccrs.PlateCarree())

    # Plot Location
    ax.scatter(loc[0],loc[1], color="r", zorder=999)

    # Plot Gridlines of Cell
    ax.vlines(range(-180, 181, 5), -90,90, color = "r")
    ax.hlines(range(-90, 91, 5), -180,180, color = "r")

    # Define Objects, colors and dataset name depending on objects given
    objects = [obj1]
    colors = ['b']
    names = [obj1.path.split('/')[-2]]
    if obj2 is not None:
        objects.append(obj2)
        colors.append('m')
        names.append(obj2.path.split('/')[-2])
    if obj3 is not None:
        objects.append(obj3)
        colors.append('c')
        names.append(obj3.path.split('/')[-2])
    if obj4 is not None:
        objects.append(obj4)
        colors.append('g')
        names.append(obj4.path.split('/')[-2])

    # Define List for necessery variables
    scatter_lats = []
    scatter_lons = []
    scatter_labels = []
    scatter_dist = []
    scatter_cells = []

    # Fill lists for each object
    for obj, color, name in zip(objects, colors, names):
    
        # Get gpi, distance, lons, lats and cellID of the surrounding gpis
        labels, dist = obj.grid.find_k_nearest_gpi(loc[0], loc[1], k=k)
        lons, lats = obj.grid.gpi2lonlat(labels)
        cells = obj.grid.gpi2cell(labels)

        # Fill lists
        scatter_lats.append(lats)
        scatter_lons.append(lons)
        scatter_labels.append(labels[0])
        scatter_dist.append(dist[0])
        scatter_cells.append(cells[0])
    
        # Plot the gpis
        ax.scatter(lons,lats, marker = "x", color = color, alpha = 0.35, zorder=998, label=name)

        # Highlight closest gpi to location
        gpi = obj.grid.find_nearest_gpi(loc[0], loc[1])
        gpi_lon, gpi_lat = obj.grid.gpi2lonlat(gpi)
        ax.scatter(gpi_lon, gpi_lat, marker = "x", color = color, alpha = 1, zorder=999)

    # Flatten the lists
    scatter_lats = [item for sublist in scatter_lats for item in sublist]
    scatter_lons = [item for sublist in scatter_lons for item in sublist]
    scatter_labels = [item for sublist in scatter_labels for item in sublist]
    scatter_dist = [item for sublist in scatter_dist for item in sublist]
    scatter_cells = [item for sublist in scatter_cells for item in sublist]

    # Plot the flattened gpis and add hovering feature
    scatter = ax.scatter(scatter_lons, scatter_lats, marker = "x", alpha=0)
    cursor = mplcursors.cursor(scatter, hover=True)
    cursor.connect("add", lambda sel: sel.annotation.set_text(f"gpi: {scatter_labels[sel.target.index]}\ncell: {scatter_cells[sel.target.index]}\ndistance: {round(scatter_dist[sel.target.index]/1000, 2)}km"))
    
    # Add a title and labels
    plt.title('Grid point Indexes')
    plt.xlabel("Longitude [°]")
    plt.ylabel("Latitude [°]")
    plt.legend(loc='upper left', title='Datasets')
    
    # Show the plot
    plt.show()

def plot_ts(data, params, timeperiod, title, fname=None):
    """
    Plot the timeseries from a dataframe
    ----------
    Parameters:
    data : pd.Dataframe
        Dataframe from which data is plotted
    params : list
        Columns of Dataframe which should be plotted
    timeperiod : list
        Start datetime and end datetime for the timeseries
    title : str
        Title of the figure
    fname : str, optional
        Filename of the saved png
    """

    # If only one parameter should be plotted:
    if len(params) == 1:
        fig, axes = plt.subplots(len(params),1)
        axes.set_title(title)
        axes.set_xlabel("Date")
        axes.plot(data[params[0]])
        axes.set_ylabel(params[0])
        axes.set_xlim(timeperiod)
        plt.setp(axes.get_xticklabels(), rotation=45) #Rotate the x-tick labels
        fig.tight_layout()

    # If more than one parameter should be plotted
    else:
        fig, axes = plt.subplots(len(params),1)
        axes[0].set_title(title)
        axes[-1].set_xlabel("Date")
        for i, (ax, param) in enumerate(zip(axes, params)):
            ax.plot(data[param])
            ax.set_ylabel(param)
            ax.set_xlim(timeperiod)
            if i == len(params) - 1:
                    plt.setp(ax.get_xticklabels(), rotation=45) # Only rotate the x-tick labels of the lowest plot
            else:
                ax.xaxis.set_visible(False) # Only show x-tick labels of the lowest plot
        fig.tight_layout()

    # Save the plot if there is a filename given
    if fname is not None:
        fig.savefig(fname)

    plt.show()
    