# Data
Description of the dataset's content, its content, format and naming conventions.


## Chemical species

| BDAP id     | ECMWF ADS id                  | BDAP description (Concentration de)                          |
| ----------- | ----------------------------- | ------------------------------------------------------------ |
| O3_USI      | ozone                         | Ozone                                                        |
| CO_USI      | carbon_monoxide               | Monoxyde de carbone                                          |
| NO2_USI     | nitrogen_dioxide              | Dioxyde d'azote                                              |
| SO2_USI     | sulphur_dioxide               | Dioxyde de soufre                                            |
| NO_USI      | nitrogen_monoxide             | Monoxyde d'azote                                             |
| PM25_USI    | particulate_matter_2.5um      | PM25                                                         |
| PM10_USI    | particulate_matter_10um       | PM10                                                         |
| NH3_USI     | ammonia                       | Ammoniac                                                     |
| NMVOC_USI   | non_methane_vocs              | Composants organiques volatils hors méthane                  |
| PANS_USI    | peroxyacyl_nitrates           | Famille des peroxyl-acétyl-nitrate                           |
| SIA_USI     | secondary_inorganic_aerosol   | Aérosols inorganiques secondaires (Sulfate-Nitrate-Ammonica) |
| DUST_USI    | dust                          | Poussières désertiques                                       |
| PM_WF_USI   | pm10_wildfires                | Concentration massique en traceur de feux biogéniques        |
| EC_TOT_USI  | total_elementary_carbon       | Carbone élémentaire total                                    |
| EC_RES_USI  | residential_elementary_carbon | Carbone élémentaire résiduel                                 |
| HCHO_USI    | formaldehyde                  | Formaldéhyde                                                 |
| CHOCHO_USI  | glyoxal                       | glyoxal                                                      |
| DYNSAL_USI  | pm10_sea_salt_dry             | Aérosols marins                                              |
| PM25_OM_USI | pm2.5_total_organic_matter    | PM25 en matière organique                                    |
| NO3_DRY_USI |                               | Aérosol secondaire de nitrate dans les PM25                  |
| NH4_DRY_USI |                               | Aérosol secondaire d’ammonium dans les PM25                  |
| SO4_DRY_USI |                               | Aérosol secondaire de sulfate dans les PM25                  |

## Levels

| level (m) | BDAP level_type |
| --------- | --------------- |
| 0         | SOL             |
| 50        | HAUTEUR         |
| 100       | HAUTEUR         |
| 250       | HAUTEUR         |
| 500       | HAUTEUR         |
| 750       | HAUTEUR         |
| 1000      | HAUTEUR         |
| 2000      | HAUTEUR         |
| 3000      | HAUTEUR         |
| 5000      | HAUTEUR         |

> ⚠️ The 11 european CTM model's level 0 (or SOL levels)  altitude are not unified.
> Depending on the model, this first surface level can represent the state
> of the atmosphere at altitude 0, 10 or 20 meters.

## Lead times
The 11 European CTM models are executed each night at midnight, producing forecasts that cover the next 96 hours. For this project we will use three specific terms to describe the timing of those forecasts:

- **`run_date`**: the calendar date (always at 00:00 hours) on which a forecast is generated.
- **`leadtime`**: the elapsed time after the run_date that must pass before the forecast becomes valid; it designates a single hour for which the forecast is applicable (e.g., a leadtime of 6 hours means the forecast is valid only for the hour that starts 6 hours after the run_date).
- **`valid_date`**: the exact date and hour when the forecast applies; it is obtained by adding the leadtime to the run_date.

In other words, the `leadtime=39h` of a model run `run_date=2024‑06‑01:00h00 UTC` is valid for the hour starting at `valid_date=2024‑06‑02 15:00 UTC`.

> 🕛 This project consistently use the UTC timezone.