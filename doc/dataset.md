# Dataset

The project's data is stored in the **dataset directory**, whose location is
given by the `CAMS_DATASET_DIR` constant in the
[`cams/settings.py`](../cams/settings.py) file (default
`/scratch/shared/cams-dl-ensemble/`):

```txt
CAMS_DATASET_DIR
└── raw
    ├── chimere
    ├── dehm
    ├── emep
    ├── euradim
    ├── gemaq
    ├── lotos
    ├── match
    ├── minni
    ├── mocage
    ├── monarch
    ├── silam
    ├── reanalysis
    └── weighted_ensemble
```

The 11 European CTM model directories contain the model forecast **input**
data, one consolidated NetCDF file per run date. The `reanalysis` and
`weighted_ensemble` directories contain the **target** data.

## Model forecasts (input)

Each of the 11 CTM models has its own directory, named after the model in
lowercase:

```
chimere, dehm, emep, euradim, gemaq, lotos, match, minni, mocage, monarch, silam
```

Each directory contains one NetCDF file per run date, gathering all requests
for that date into a single file:

```txt
<MODEL_DIR>/YYYY_MM_DD-<SPECIES>-<LEVEL>m-<LT_START>-<LT_END>h.netcdf
```

For example `chimere/2023_08_30-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.netcdf`
contains the CHIMERE forecasts produced on `2023-08-30`, for the species
`CO, NO2, PM10, PM25, SO2, O3`, at the surface level (0 m), for every lead
time from 0 to 96 hours.

The files hold one data variable per species (e.g. `co_conc`, `no2_conc`,
`o3_conc`, `pm10_conc`, `pm2p5_conc`, `so2_conc`) with dimensions
`(time, level, latitude, longitude)` and values in µg/m³.

A request that failed to download is stored as a `.fail` file with the same
stem (e.g. `2023_01_01-CO_NO2_PM10_PM25_SO2_O3-0m-0-96h.fail`). Successfully
downloaded requests appear either as `.netcdf` files or, in a few cases, as
`.zip` archives with the same stem.

## Reanalysis (target)

The `reanalysis` directory contains the CAMS European air quality reanalysis
data, split into one NetCDF file per species and per month:

```txt
reanalysis/cams.eaq.<TYPE>.ENSa.<SPECIES>.l0.YYYY-MM.nc
```

- **`TYPE`** = either `ira` (interim reanalysis) or `vra` (validated
    reanalysis).
- **`SPECIES`** = the species, in lowercase (`co`, `no2`, `o3`, `pm10`,
    `pm2p5`, `so2`).
- **`l0`** = surface level (0 m).
- **`YYYY-MM`** = the month covered by the file.

For example `reanalysis/cams.eaq.ira.ENSa.o3.l0.2025-01.nc` contains the
hourly O3 interim reanalysis for January 2025. Each file holds a single data
variable named after the species (e.g. `o3`) with dimensions
`(time, lat, lon)` and values in µg/m³.

The raw download can also be stored as a per-month `.zip` archive
(`-YYYY-MM-DD-CO_NO2_PM10_PM25_SO2_O3-0m-<TYPE>.zip`), and failed requests
as `.fail` files with the same stem.

## Weighted ensemble (target)

The `weighted_ensemble` directory contains the weighted ensemble data, split
into one GRIB file per run date and per species:

```txt
weighted_ensemble/YYYY_MM_DD-<SPECIES>-0m-0-96h.grib
```

For example
`weighted_ensemble/2025_10_10-O3-0m-0-96h.grib` contains the O3 weighted
ensemble forecast run on `2025-10-10`, at the surface level, for every lead
time from 0 to 96 hours. The available species are `CO`, `NO2`, `O3`, `PM10`,
`PM25`, `SO2`. Failed requests are stored as `.fail` files with the same stem.

## Naming conventions

- **`YYYY_MM_DD`** = the run date (when the forecast is generated), ordered
  year, month then day, `_` separated and zero padded on the left.
- **`<SPECIES>`** = one or several species, `_` separated and ordered in the
  usual project order (`CO_NO2_PM10_PM25_SO2_O3` for the model files, a single
  species for the weighted ensemble files).
- **`<LEVEL>m`** = the height of the level in meters (`0m` for the surface).
- **`<LT_START>-<LT_END>h`** = the range of forecast lead times in hours
  covered by the file (typically `0-96h`).
