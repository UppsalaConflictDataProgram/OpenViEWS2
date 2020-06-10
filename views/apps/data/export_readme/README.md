# Views Tables and Geoms

This is a data export from the ViEWS project.
Python code is available for joining these files into usable datasets at priogrid-month and country-month level and computing a large set of transformations on them.
See https://github.com/UppsalaConflictDataProgram/OpenViEWS2 for instructions on how to get started.

If you don't wish to use the python tooling but instead prepare your own data, read on.
There are three types of files here:

* Skeleton tables, that represent a level of analysis in ViEWS and hold identifiers.
* Data tables, that hold imputed source data at their native level of analysis.
* Geometries in .geojson format to enable plotting and spatial transformations.

The key identifiers are

* country_id, that correspond to ids from cshapes.
* pg_id, PRIO-GRID ids.
* year
* month_id, an incremental month identifier where 1 is 1980-01

Skeletons are available for

* priogrid-month (pgm)
* priogrid-year (pgy)
* country-month (cm)
* country-year (cy)

Included data sources are

* ACLED, from https://www.acleddata.com/
* FVP, a custom dataset of CY data.
* GED from the UCDP
* pgdata, from PRIOGRID
* REIGN, from https://oefdatascience.github.io/REIGN.github.io
* SPEI, from https://spei.csic.es/map/maps.html
* VDEM, from https://www.v-dem.net/en/
* WDI, from http://datatopics.worldbank.org/world-development-indicators/

Data tables containing "\_imp\_skelarn_number" are imputed using scikit learns in 5 different imputations and should have no missingness in numeric columns.
Data tables ending in \_part\_number are partitioned to work around column number limitations in our database.

To construct a usable dataset from these start with a skeleton table and then join in the data sources that you want.