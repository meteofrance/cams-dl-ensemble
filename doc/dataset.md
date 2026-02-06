# Dataset

## Raw data

A large amount of raw meteorological data is necessary to train the cams deeplearning ensemble model. Here is a description of the data that should be gathered, and how to organize it so that it can be preprocessed for training.

This project's preprocessing script expects data organized like so, with the following naming conventions:
- **`YYYY_MM_DD`** specifies a date ordered with year, month then day, spearated with `_` characters and zero padded on the left.
- **`LT`** = leadtime, a zero paded number between 0 and 96.
- **`LVL`** = level, one of 0, 50, 100, 250, 500, 750, 1000, 2000, 3000, 5000.
- **`SPECIESID`** = a species BDAP id such as specified [here](dataset.md), without the suffix `_USI`.
```txt
.
├── reanalisis
│   └── SPECIESID
│       └── YYYY_MM_LVLm.netcdf
├── PMACCCHIMERE
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCDEHM
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCEMEP
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCEURADIM
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCGEMAQ
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCLOTOS
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCMATCH
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCMINNI
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCMOCAGE
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
├── PMACCMONARCH
│   └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
└── PMACCSILAM
    └── YYYY_MM_DD_LT_LVL_SPECIESID.grib
```

Files in folders named with a CTM model name contain input data for 1 model run, 1 leadtime,
1 species and all levels. They are `.grib files that can be opened using python like so:
```py
>>> import xarray as xr
>>> dataarray = xr.open_dataarray("PMACCCHIMERE/2023_07_27_15_O3.grib")
>>> dataarray
<xarray.DataArray 'unknown' (latitude: 420, longitude: 700)> Size: 1MB
[294000 values with dtype=float32]
Coordinates:
  * latitude    (latitude) float64 3kB 71.95 71.85 71.75 ... 30.25 30.15 30.05
  * longitude   (longitude) float64 6kB -24.95 -24.85 -24.75 ... 44.85 44.95
    time        datetime64[ns] 8B ...
    step        timedelta64[ns] 8B ...
    surface     float64 8B ...
    valid_time  datetime64[ns] 8B ...
Attributes: (12/30)
    GRIB_paramId:                             0
    GRIB_dataType:                            fc
    GRIB_numberOfPoints:                      294000
    GRIB_typeOfLevel:                         surface
    GRIB_stepUnits:                           1
    GRIB_stepType:                            instant
    ...                                       ...
    GRIB_name:                                unknown
    GRIB_shortName:                           unknown
    GRIB_units:                               unknown
    long_name:                                unknown
    units:                                    unknown
    standard_name:                            unknown
```

Files in the `reanalisis` folder contain the reanalisis target data,
1 species, 1 month and one level. They are `.netcdf` files that can be opened
using python like so:
```py
>>> import xarray as xr
>>> import earthkit.data as ekd
>>> source = ekd.from_source("file", "reanalisis/O3_USI/2022_05_0m.netcdf")
>>> data_array: xr.DataArray = source.to_xarray().to_dataarray()[0]
>>> data_array
<xarray.DataArray (time: 744, lat: 420, lon: 700)> Size: 875MB
dask.array<getitem, shape=(744, 420, 700), dtype=float32, chunksize=(744, 420, 700), chunktype=numpy.ndarray>
Coordinates:
  * time      (time) datetime64[ns] 6kB 2022-05-01 ... 2022-05-31T23:00:00
  * lat       (lat) float64 3kB 30.05 30.15 30.25 30.35 ... 71.75 71.85 71.95
  * lon       (lon) float64 6kB -24.95 -24.85 -24.75 ... 44.75 44.85 44.95
    variable  <U2 8B 'o3'
Attributes:
    Conventions:  CF-1.7
    Title:        CAMS European air quality validated reanalysis
    Provider:     COPERNICUS European air quality service
    Production:   COPERNICUS Atmosphere Monitoring Servic
```

## Processed data
The raw data need to be processed before being usable for training.
Data processing can be done with:
```sh
python scripts/data_processing.py \
    --raw_folder <path/to/raw/folder> \
    --output_folder <path/to/output/folder>
```

Once processing done, training ready data are ordered like so:
```txt
.
├── input
│  └── YYYY_MM_DD.netcdf
└── target
   └── YYYY_MM_DD_HH.netcdf
```

Processed data can be opened like so:
```py
>>> import xarray as xr
>>> input_dataarray = xr.open_dataarray("processed_data/input/2023_12_18.netcdf")
>>> input_dataarray
<xarray.DataArray 'unknown' (model: 11, species: 1, level: 1, leadtime: 1,
                             latitude: 420, longitude: 700)> Size: 13MB
[3234000 values with dtype=float32]
Coordinates:
  * model       (model) <U12 528B 'PMACCCHIMERE' 'PMACCDEHM' ... 'PMACCSILAM'
  * species     (species) <U2 8B 'O3'
  * level       (level) <U1 4B '0'
  * leadtime    (leadtime) <U2 8B '15'
  * latitude    (latitude) float64 3kB 71.95 71.85 71.75 ... 30.25 30.15 30.05
  * longitude   (longitude) float64 6kB -24.95 -24.85 -24.75 ... 44.85 44.95
    time        datetime64[ns] 8B ...
    step        timedelta64[ns] 8B ...
    surface     float64 8B ...
    valid_time  datetime64[ns] 8B ...
Attributes: (12/30)
    GRIB_paramId:                             0
    GRIB_dataType:                            fc
    GRIB_numberOfPoints:                      294000
    GRIB_typeOfLevel:                         surface
    GRIB_stepUnits:                           1
    GRIB_stepType:                            instant
    ...                                       ...
    GRIB_name:                                unknown
    GRIB_shortName:                           unknown
    GRIB_units:                               unknown
    long_name:                                unknown
    units:                                    unknown
    standard_name:                            unknown
>>> target_dataarray = xr.open_dataarray("processed_data/target/2023_12_21_15.netcdf")
>>> target_dataarray
<xarray.DataArray 'O3' (lat: 420, lon: 700)> Size: 1MB
[294000 values with dtype=float32]
Coordinates:
  * lat       (lat) float64 3kB 71.95 71.85 71.75 71.65 ... 30.25 30.15 30.05
  * lon       (lon) float64 6kB -24.95 -24.85 -24.75 ... 44.75 44.85 44.95
    time      datetime64[ns] 8B ...
    variable  <U2 8B ...
Attributes:
    Conventions:  CF-1.7
    Title:        CAMS European air quality interim reanalysis
    Provider:     COPERNICUS European air quality service
    Production:   COPERNICUS Atmosphere Monitoring Service
    units:        µg/m3
```

Processed data can be represented with the `plot.py` script:
```sh
python scripts/plot_sample.py --date YYYYMMDDLT
```

![O3 concentration plot for the 11 models, median ensemble and VRA model runs of 01/04/2023 valid time 15h](images/2023_04_01_exemple.png)
