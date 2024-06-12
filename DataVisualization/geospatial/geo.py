import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load your geospatial data (replace 'filename.shp' with your actual file path)
world = gpd.read_file('ne_10m_admin_0_sovereignty.shp')

# Print column names to confirm the available columns
print(world.columns)

# Convert 'SOVEREIGNT' column to numeric representation
sovereignt_map = {value: index for index, value in enumerate(world['SOVEREIGNT'].unique())}
world['SOVEREIGNT_NUM'] = world['SOVEREIGNT'].map(sovereignt_map)

# Plotting
fig, ax = plt.subplots(1, figsize=(16, 8), facecolor='lightblue')

world.plot(ax=ax, color='black')

# Plotting based on sovereignty status
world.plot(ax=ax, column='SOVEREIGNT_NUM', cmap='Reds', edgecolors='grey')

# axis for the color bar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="3%", pad=0.05)

# color bar
vmax = world['SOVEREIGNT_NUM'].max()
mappable = plt.cm.ScalarMappable(cmap='Reds', norm=plt.Normalize(vmin=0, vmax=vmax))
cbar = fig.colorbar(mappable, cax=cax)

ax.axis('off')
plt.show()





