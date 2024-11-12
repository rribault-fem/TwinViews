import xarray as xr
import matplotlib.pyplot as plt



def plot_envir_histograms_matplotlib(dataset: xr.Dataset) -> plt.Figure:

    fig, ax = plt.subplots(2,2)
    bar_color = 'cornflowerblue'
    density = False

    # Get mean 1 minute wind speed over the whole dataset
    wind_1min = dataset['AI_WindSpeed'].dropna(dim='time_sensor', how='all').rolling(time_sensor=60).mean()
    wind_1min.plot.hist(ax=ax[0,0], bins=20, color=bar_color, density=density, yscale='log')
    ax[0,0].set_ylabel('count')
    ax[0,0].set_xlabel('Wind speed [m/s]')

    # Get mean 1 minute wind speed over the whole dataset
    power_1min = dataset['V_GridRealPowerLog'].dropna(dim='time_sensor', how='all').rolling(time_sensor=60).mean()
    power_1min.plot.hist(ax=ax[0,1], bins=20, color=bar_color, density=density, yscale='log')
    ax[0,1].set_ylabel('count')
    ax[0,1].set_xlabel('Turbine power [kw]')

    # get current speed
    dataset['simu_cur'].plot.hist(ax=ax[1,1], bins=20, color=bar_color, density=density, yscale='log')
    ax[1,1].set_ylabel('count')
    ax[1,1].set_xlabel('Current speed [m/s]')

    # Get wave Hs
    dataset['simu_hs'].plot.hist(ax=ax[1,0], bins=20, color=bar_color, density=density, yscale='log')
    ax[1,0].set_ylabel('count')
    ax[1,0].set_xlabel('Wave : Hs [m]')


    # Add a general title to the figure
    fig.suptitle('Histograms of Environmental and Power output Data', fontsize=16)

    # Adjust layout to make room for the title
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.tight_layout()
    return fig