"""
ldcheck: check the shape of REMA strip DEMs against Landsat 8/9 panchromatic images.

Modules
-------
landsat   : STAC search, USGS ERS login, windowed (COG) reads of B8 and angle bands
dem       : find and read 2-m strip DEMs (+matchtag, bitmask) onto a 3-m working grid
shading   : predicted illumination (cos incidence + cast shadows) on the Landsat 15-m grid
velocity  : velocity fields for advecting images to the DEM time
stack     : build the advected image stack, register it, find clouds
compare   : local DEM-vs-image statistics, classification, per-DEM summary
edit      : combine standard / relaxed dem_filter output with the Landsat classification
check_dem : the per-DEM driver
"""
