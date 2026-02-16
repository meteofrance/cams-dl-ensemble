# Data
Description of the dataset's content, its content, format and naming conventions.


## Chemical species

| MF database id | ECMWF ADS id                  | BDAP description (Concentration de)                   |
| -------------- | ----------------------------- | ----------------------------------------------------- |
| O3_USI         | ozone                         | Ozone                                                 |
| CO_USI         | carbon_monoxide               | Carbon monoxide                                       |
| NO2_USI        | nitrogen_dioxide              | Nitrogen dioxide                                      |
| SO2_USI        | sulphur_dioxide               | Sulfur dioxide                                        |
| NO_USI         | nitrogen_monoxide             | Nitrogen monoxide                                     |
| PM25_USI       | particulate_matter_2.5um      | PM2.5 (particulate matter ≤ 2.5 µm)                   |
| PM10_USI       | particulate_matter_10um       | PM10 (particulate matter ≤ 10 µm)                     |
| NH3_USI        | ammonia                       | Ammonia                                               |
| NMVOC_USI      | non_methane_vocs              | Non‑methane volatile organic compounds (NMVOCs)       |
| PANS_USI       | peroxyacyl_nitrates           | Peroxyacetyl nitrate family                           |
| SIA_USI        | secondary_inorganic_aerosol   | Secondary inorganic aerosol (sulfate‑nitrate‑ammonia) |
| DUST_USI       | dust                          | Desert dust                                           |
| PM_WF_USI      | pm10_wildfires                | Mass concentration of biogenic fire tracer            |
| EC_TOT_USI     | total_elementary_carbon       | Elemental carbon total                                |
| EC_RES_USI     | residential_elementary_carbon | Residual elemental carbon                             |
| HCHO_USI       | formaldehyde                  | Formaldehyde                                          |
| CHOCHO_USI     | glyoxal                       | Glyoxal                                               |
| DYNSAL_USI     | pm10_sea_salt_dry             | Marine aerosols (sea‑salt)                            |
| PM25_OM_USI    | pm2.5_total_organic_matter    | PM2.5 organic matter                                  |
| NO3_DRY_USI    |                               | Nitrate secondary aerosol in PM2.5                    |
| NH4_DRY_USI    |                               | Ammonium secondary aerosol in PM2.5                   |
| SO4_DRY_USI    |                               | Sulfate secondary aerosol in PM2.5                    |

> ⚠️ `NO3_DRY_USI`, `NH4_DRY_USI` and `SO4_DRY_USI` data exist in Météo France's
> database for the 11 CTM models, but no reanalisis for it exist yet.

## Levels

| level (m) | level type (ground or height) |
| --------- | ----------------------------- |
| 0         | SOL                           |
| 50        | HAUTEUR                       |
| 100       | HAUTEUR                       |
| 250       | HAUTEUR                       |
| 500       | HAUTEUR                       |
| 750       | HAUTEUR                       |
| 1000      | HAUTEUR                       |
| 2000      | HAUTEUR                       |
| 3000      | HAUTEUR                       |
| 5000      | HAUTEUR                       |

> ⚠️ The 11 european CTM model's level 0 (or SOL levels)  altitude are not unified.
> Depending on the model, this first surface level can represent the state
> of the atmosphere at altitude 0, 10 or 20 meters. In this project all the
> surface levels are considered to be at altitude 0 meter.

## Lead times
The 11 European CTM models are executed each night at midnight, producing forecasts that cover the next 96 hours. For this project we will use three specific terms to describe the timing of those forecasts:

- **`run_date`**: the calendar date (always at 00:00 hours) on which a forecast is generated.
- **`leadtime`**: the elapsed time after the run_date that must pass before the forecast becomes valid; it designates a single hour for which the forecast is applicable (e.g., a leadtime of 6 hours means the forecast is valid only for the hour that starts 6 hours after the run_date).
- **`valid_date`**: the exact date and hour when the forecast applies; it is obtained by adding the leadtime to the run_date.

In other words, the `leadtime=39h` of a model run `run_date=2024‑06‑01:00h00 UTC` is valid for the hour starting at `valid_date=2024‑06‑02 15:00 UTC`.

> 🕛 This project consistently uses the UTC timezone.

## Grids

Not all the european CTM models have the same the same grid !:
The median ensemble, MOCAGE CHIMERE EURADIM EMEP MATCH DEHM GEMAQ MINNI and
MONARCH use the following latitude and longitudes (extract):
```txt
latitude  [71.95 71.85 71.75 71.65 71.55 71.45 ...]
longitude [-24.95 -24.85 -24.75 -24.65 -24.55 -24.45 ...]
```

While LOTOS and SILAM use:
```txt
latitude  [71.949997 71.849997 71.749997 71.649997 71.549997 71.449997]
longitude [-24.950001   -24.850001   -24.75000099 -24.65000099 -24.55000099 -24.45000099]
```

During preprocessing of the raw dataset, we do not interpolate the grid points,
and accept the slight error introduced (0.33 meters).