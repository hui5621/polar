# Environment data sources

`data/environments/projected_nsr_2025/` contains twelve frozen, 325×209 projected raster snapshots for 2025-08-05 through 2025-10-12. The `.npz` files are processed environment grids, not the original source files. Original NetCDF, Shapefile, GeoTIFF, ENC packages, chart images, reference routes, raw AIS tracks, and local absolute paths are excluded.

The packed layers include sea-ice class, concentration and thickness; ocean-current components and speed; land, ice and chart blocking masks; chart-risk values; bathymetry; and a rasterized route-corridor score. The manifest lists the scene dates, projection, grid spacing and start/goal cells.

The saved 2025-10-05 environment has all-zero `current_u`, `current_v` and `current_speed` layers. Its scene metadata records no CMEMS input for that date, so these zero arrays represent missing current data rather than a measured zero-current field. The stored map is preserved as provided; the planners do not add replacement values.

## Attribution

- Arctic ocean currents and model fields: **Arctic Ocean Physics Analysis and Forecast**, Copernicus Marine product `ARCTIC_ANALYSISFORECAST_PHY_002_001`, daily dataset `cmems_mod_arc_phy_anfc_6km_detided_P1D-m`, DOI [10.48670/moi-00001](https://doi.org/10.48670/moi-00001). Copernicus Marine asks users distributing derivative products to credit the service and include the product DOI; see its [citation guidance](https://help.marine.copernicus.eu/en/articles/4444611-citing-copernicus-marine-products-and-services) and [product page](https://data.marine.copernicus.eu/product/ARCTIC_ANALYSISFORECAST_PHY_002_001/description).
- Bathymetry: **GEBCO_2025 Grid**. GEBCO permits publishing and adapting its grid-derived products, with source acknowledgement. See [GEBCO terms of use](https://www.gebco.net/data-products/gridded-bathymetry/terms-of-use).
- Sea-ice chart classes: U.S. National Ice Center (USNIC) Arctic Marginal Ice Zone / sea-ice chart products, listed in the [USNIC product catalog](https://usicecenter.gov/Catalog).
- Chart-risk and route-corridor layers: processed raster values derived from the saved bathymetry and corridor representation. Original AIS tracks and point routes are not included.

The saved arrays are intended only for offline algorithm research. GEBCO explicitly says its grid should not be used for navigation or other safety-at-sea purposes.
