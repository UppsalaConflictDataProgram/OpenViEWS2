Manual
=======

Introduction
-------------

Welcome to the ViEWS manual.

This document will help get you started using ViEWS code and data.

This is very much a work in progress, hence the scattered (REFERNCE) markers and dodgy formatting, feel free to contact the team for any clarifications while we figure out this documenting system.
We apologise for any omitted or incorrect citations, we are working on them.

If you are in a hurry, don't want to use ViEWS python code and are just looking for data in plain old csv head to https://views.pcr.uu.se/download/datasets/ and grab the latest timestamped .zip from there.

If you're looking for the code it can be found here: https://github.com/UppsalaConflictDataProgram/OpenViEWS2

For the main website see https://views.pcr.uu.se

The current state of this document is a mix between a gentle introduction and a development manual.
The ambition is to split it into two, one more direct "Getting started tutorial" kind of document and one more in-depth technical manual.

This document assumes

* Basic familiarity with the terminal / shell
* Basic familiarity with python


Installation and hardware
--------------------------

ViEWS is implemented mainly in Python.

Some parts of ViEWS require 32GB of RAM or more as data sizes can be fairly large.
If you are stuck because you hit memory limits contact the team and we might be able to help you with a workaround.

To get started you must first install some dependencies.

Mac / Linux
~~~~~~~~~~~

To get started as a Mac or Linux user follow these steps:

Install the miniconda python distribution from https://docs.conda.io/en/latest/miniconda.html . You want the 3.7 distribution.
When prompted answer yes to activate conda for your shell.
After fininshing the installation your terminal should have a prefix saying ``(base)`` on your terminal prompt.
You might need to open a new terminal for changes to take effect.
After conda is installed, run the following in your terminal

.. code-block:: bash

    git clone https://github.com/UppsalaConflictDataProgram/OpenViEWS2
    cd OpenViEWS2
    ./install_views2.sh
    conda activate views2

You should now have a working installation of the `views` package in the `views2` conda environment.
You can now proceed to fetching data.

Windows
~~~~~~~

We currently do not have a windows environment to test or build on so these steps are not tested.
However you should be able to get going by:

* Getting the code from github.
* Create the conda environment from the ``misc/environment.yaml`` file
* Activate the ``views2`` conda environment and run ``python setup.py``.
* Copy the default ``config.yaml`` from ``misc/config.yaml`` to the root of the repository.

The ``views2`` conda environment should now be created, however you may run into issues. See the conda installation documentation at https://conda.io/projects/conda/en/latest/user-guide/install/index.html . Feel free to contact the team if you get stuck.


Configuration (ViEWS team)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some configuration is necessary if you need database access or to use supercomputing resources through slurm.
If you are a member of the ViEWS team and have database access you will need to set your database username in the config.yaml file in the root of the repository.
In config.yaml update

* databases: views: user: change `username # CHANGE ME!` into your actual database username.
* databases: views: host: into the hostname or ip-address of the views database server.

Then make sure you have up-to-date certificates in the locations pointed in the config file, contact MC to get these.

If you are a member of the public without database access you should be able to use almost all ViEWS functionality without database access, see the Data section (REFERENCE TODO).

If you need to run things in a supercomputing environmnent that uses slurm  (such as uppmax) you will need to add your slurm username and project to the config.yaml also.


Running code
------------

There are two main ways to run ViEWS python code, through notebooks or scripts.

Notebooks - Jupyter Lab
~~~~~~~~~~~~~~~~~~~~~~~~

Notebooks through Jupyter Lab are a convenient way to write analysis or development code.
Jupyter Lab is installed alongside the ``views`` package when running ``./install_views2.sh`` so no extra setup is necessary.
To start a jupyter lab server on your local machine run the following in your terminal:

.. code-block:: bash

    conda activate views2
    jupyter lab

You now have a jupyter lab server running in the ``views2`` conda environment.
Your browser should open and show you the interface.
Have a look around the projects directory and find a notebook to look at or create a new one.
``projects/model_development/example.ipynb`` is a good place to start looking.

Note that jupyter lab has replaced jupyter notebook as the primary platform for notebook development.
It provides many new features and a nicer interface.
The old ``jupyter notebook`` command is still available of course if you don't like change.

Scripts
~~~~~~~

Running scripts is also done through the `views2` environment:

.. code-block:: bash

    conda activate views2
    python path/to/myscript.py

This will run the myscript.py in the ``views2`` python environment where the ``views`` package is available.
ViEWS has a convention that scripts should go either in the /runners/ directory, for core ViEWS functionality like the prediction pipeline, or in the /projects/ directory where more miscellaneous work happens.
Please don't put scripts that have side effects (actually do things) without an

.. code-block:: python

    if __name__ == "__main__":
        do_stuff()

block in the ``views`` directory.


The ``views`` python package
-----------------------------


Views code is organised as a python package called ``views``.
It is not available from repositories like pip or conda, it must be installed from source. If you  completed the installation you have it.
In ``python`` in the ``views2`` environment after installation, you can import it like so

.. code-block:: python

    import views


For example usage start a ``jupyter lab`` server and see ``projects/model_development/example.ipynb`` or ``projects/prediction_competition/benchmark_notebook.ipynb``.

Doing ``import views `` currently takes a couple of seconds as model specifications are solved.
We are working on speeding that up, no-one likes waiting!

Logging
~~~~~~~~

It's nice to know what's going on when you run a command.
ViEWS has a lot of functionality "under the hood" which you might want to see, especially if something seems slow or wrong.
A nice preamble to your python scripts that sets up logging is:

.. code-block:: python

    import logging
    import views

    logging.basicConfig(
        level=logging.DEBUG,
        format=views.config.LOGFMT,
        handlers=[
            logging.StreamHandler(),
        ],
    )
    log = logging.getLogger(__name__)

    log.info("Hello!")

If this is too verbose for you feel free to change ``logging.DEBUG`` to ``logging.INFO``.

If your script uses ``multiprocessing`` and appears to get stuck this might be related to this preamble, especially the StreamHandler() part.
We are investigating this. (TODO)

Conventions
~~~~~~~~~~~~

``df`` is short for dataframe and is the standard name for pandas dataframes.
``s`` i short for series and is the standard name for a pandas series, which is a single column data representation.

Dataframes should be assigned an index whenever it isn't crazy to do so.
The levels set should be ``[timevar, groupvar]``, where ``timevar`` is usually ``"month_id"`` and groupvar is usually ``"pg_id"`` or ``"country_id"`` .
This is so that ``df.loc[time]`` gets a temporal subset of df.
Similarly, ``df.groupby(level=1)`` is often used to implement temporal transformations by group.
ViEWS data is almost always a panel so this convention makes life a bit easier and much core code relies on it.

Data
-----

The primary ViEWS working data is hosted on an internally accessible database server, called janus.
For the public we export a .zip of .csv files at
`https://views.pcr.uu.se/download/datasets/ <https://views.pcr.uu.se/download/datasets/>`_ .
These should reflect the internal working data as closely as possible.
The latest (and recommended) public data exports are prefix ``views_tables_and_geoms_{date}.zip``.

Views uses data from many sources.
We have gone through a many iterations of systems to keep data

* Easy to use
* Well documented
* Correct
* Up to date
* Available without missingness
* Easy to maintain
* Publicly accessible

We hope to have finally landed on a format that is usable both by us and available to the public with as little friction as possible.

ViEWS does not aim to have the complete datasets used for analysis exactly rebuildable from sources.

PostgreSQL is the main database used by ViEWS.

For details on available data see the codebook at the end of this document (REFERENCE).

There are three ways of getting ViEWS data, two for the public and one for members of the ViEWS team.

Public data, Python (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you are a member of the public wishing to get the ViEWS data and want to use ViEWS python tooling, do this on your shell after following the installation instructions.

.. code-block:: bash

    conda activate views2
    python runners/import_data.py --fetch

This will download all the latest public data to your local storage directory,
`storage/tables` in the root directory by default.
You can now use datasets in python

.. code-block:: python

    import views

    # Show available datasets
    print(list(views.DATASETS.keys()))

    # Select one
    dataset = views.DATASETS["cm_africa_imp_0"]
    # Get the usable dataframe.
    # This will take a while as the data is assembled.
    df = dataset.df


Public data, plain .csv in .zip
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want plain .csv files and to not touch python, visit the ViEWS datasets website at
`https://views.pcr.uu.se/download/datasets/ <https://views.pcr.uu.se/download/datasets/>`_ .
There you will find .zip packaged .csv files with a timestamp.
Download the one with the latest timestamp and extract it.

To reconstruct the datasets start with one of the skeleton files and then
join in the data you want to use from the source table files.

Transformations are not available in these .csv files as we compute very
many of them and distributing them as .csv would be prohitibively expensive storage wise.

If you want the transformations in .csv files you can follow the python approach above and export to csv with

.. code-block:: python

    df.to_csv("exported_data.csv")


Users with database access (ViEWS team)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you have database access as part of ViEWS you can skip the public data fetching and drink straight from the source.
First, make sure you have the views package installed by running through the installation instructions (REFERENCE).
Next, update the config.yaml file in the root of the repository with your database username and host (REFERENCE config).
Make sure you have valid postgres certificates in the location specified in the config.yaml, ask MC for help if you need certificates.
Once these things are setup you should be able to use the datasets as in the example above, they will be fetched from the database automatically as you use them. No need to run ``import_data.py``.


Using Datasets
~~~~~~~~~~~~~~

When using ViEWS tooling datasets are provided as instances of the ``Dataset`` class from the ``views.DATASETS`` dictionary.
Datasets have a property called ``.df`` that give you the actual data as a Pandas DataFrame.
To use it do the following in python:

.. code-block:: python

    # Import the views package you installed
    import views

    # Show names of available datasets in the views.DATASETS dict
    print(views.DATASETS.keys())

    # Get a Dataset instance, choose any of the above printed names
    dataset = DATASETS["cm_global_imp_0"]

    # Get a pandas dataframe of our dataset
    df = dataset.df

    # List all cols
    for col in df:
        print(col)

    # Show us some data
    print(df.head())

Simply choose a dataset from the ``views.DATASETS`` dictionary and ask for it's ``.df``.
Make sure to have fetched the data first if you don't have database access.
For working with the data see the Pandas documentation at https://pandas.pydata.org/pandas-docs/stable/getting_started/index.html#intro-to-pandas

How does it work though?

Data Implementation
--------------------
The current setup in views data management is an iteratively built system that has evolved over several years and iterations in our database.
Work here is ongoing.

There are two main geographic units of analysis in the ViEWS system.
Priogrids, as defined by `PRIO <http://grid.prio.org/>`_.
And countries, as defined by `cshapes <http://nils.weidmann.ws/projects/cshapes.html>`_ .
`pg_id` and `country_id` are unique identifiers for these geographic units.
ViEWS works primariliy at the monthly temporal resolution, we use an id system where month_id=1 corresponds to 1980.01, January 1980.

These identifiers are defined in the staging schema of the database in the following tables:

* country
* country_month
* country_year
* month
* priogrid
* priogrid_month
* priogrid_year
* year

These tables are the source of truth for identifiers that has been used from the very first ViEWS forecasts.

The two main levels of analysis in ViEWS that we publish forecasts for are the `pgm` (`prigrid-month`) and `cm` (`country-month`).
Work is being done to add an actor layer.

ViEWS does collect some data (from surveys) but most data comes from external sources.

The core components of the current implementation are

* ``Dataset``
* ``Table``
    * Skeleton tables
    * Data tables
* ``Transform``

Let's start with Table.
There are currently two types of tables, skeleton tables and data tables.

Skeleton tables
~~~~~~~~~~~~~~~~
The skeleton tables are built from the staging tables.

There are currently 8 of them

* skeleton.cm_africa
* skeleton.cm_global
* skeleton.cy_africa
* skeleton.cy_global
* skeleton.pgm_africa
* skeleton.pgm_global
* skeleton.pgy_africa
* skeleton.pgy_global

They form the scaffold or skeleton into which the data tables are joined to create usable datasets.
The identifiers are

* ``country_id`` is the cshapes country id
* ``pg_id`` for priogrid gid
* ``month_id`` for the month id where 1 is 1980-01
* ``year`` is the year
* ``month`` is the calendar month

These can be used to join any higher resolution (priogrid, month) skeleton to any lower or same resolution data table (country, year) [#]_ .


.. [#] There is one badness in the country-time skeleton tables. In order to simplifiy the time shifting needed for step ahead predictions, dynamic simulation and several temporal transformations the country tables have been transformed into a balanced panel based on currently existing countries. This means countries that no longer exist (Pre-split Sudan, Soviet Union, Yugoslavia, etc) are not included in the public data and countries that currently exist are back-filled to years where they didn't actually exist. This is done by "last observation carried backwards" for most sources. This is an issue we are working on fixing as it does potentially have a small impact on model training. It is a difficult problem though as so much of ViEWS deals with temporal dynamics, which are complicated severely by changing sets of individuals.

In code these are defined as instances of ``views.Table``.

Data tables
~~~~~~~~~~~~

The actual data from the sources lives in tables.
Source's data live in separate database schemas, one per source.
In the distributed .zip each .csv corresponds to one table.

Sources that have more than ~1600 columns must be split into multiple tables due to limitations in PostgreSQL.
These are named ``source.skeleton_imp_impmethod_impnumber_part_partnumber``.

For example for WDI at global country-year level: ``wdi_202005.cy_unimp_part_1`` and ``wdi_202005.cy_unimp_part_2`` hold unimputed data.
``wdi_202005.cy_imp_sklearn_0_part_1``, ``wdi_202005.cy_imp_sklearn_0_part_2``, hold imputation 0.

This "interface" creates a contract between the data source and the Dataset class.

* Source tables should be identical in number of rows to a skeleton table so that inner joining them does not drop rows.
* Imputed tables must have no missingness (except GED).
* Unimputed tables may have missingness.
* Source tables must have a common column prefix, such as ``wdi_``, so that it is immediately apparent from where a column originates.
* Source tables must be created in the database by calls to ``fetch_{sourcename}()`` followed by ``load_{sourcename}()`` from ``views.database.sources.{sourcename}``
* Source tables should not lose any available columns on data updates. If a source changes their set of columns these should go to a new versioned schema so that dependent datasets and models can migrate painlessly.

Missingness is handled in two ways:

* Sources with "no missingness" (ICGCW for example, missingness is a feature instead) present a single table.
* Sources with missingness present unimputed and imputed tables separately.

Exposing unimputed tables lets us work on different imputation approaches.
Imputed tables should be 5 or 10 in number and start their counting from 0.

In python tables are represented as instances of the ``views.Table`` class.
You can get a dictionary of all defined tables by

.. code-block:: python

    import views

    # Show defined tables
    for name in views.TABLES.keys():
        print(name)

    # Select one table and get its dataframe.
    table_wdi_part_1 = views.TABLES["wdi_202005.cy_imp_sklearn_0_part_1"]
    df = table_wdi_part_1.df
    print(df.head())


Data is distributed from the database and the public data exports as tables at the source's native resolution.
This saves us from copying identical yearly data to each 12 months for a year or country level data to each priogrid in a country.

Multiple tables form a dataset.

Transformations
~~~~~~~~~~~~~~~~

TODO


``Dataset``
~~~~~~~~~~~~~

In order to use the data from different sources they must be joined together to a single level of analysis in a single python dataframe.

The Dataset class (``views.Dataset``, API-REFERENCE) provides this.

`views.DATASETS` exposes a dictionary of ``Dataset`` objects.

Datasets are defined as a

* A skeleton table
* A list of tables, that hold the actual source data
* A level of analysis ("cm", or "pgm") for defining the geometries needed for spatial transformations.
* An optional list of transformations
* An optional list of columns to subset the sources by, needed for PGM where we can't fit all data in memory on any machine.


To use a dataset in python

.. code-block:: python

    from views import DATASETS

    # Show names of available datasets
    print(DATASETS.keys())

    # Get our Dataset instance
    dataset = DATASETS["cm_global_imp_0"]

    # Get a pandas dataframe of our dataset
    df = dataset.df

The last line is where the "magic" happens.
When `dataset.df` is called, the Dataset class

* Fetches all the source data from its tables. If the tables are already cached (such as after running ``runners/import_data.py --fetch``) it just read their cached copies. If not they will be fetched from the database for users with access.
* Joins them together using the identifiers from the skeleton table.
* Computes all the transformations.
* Caches the final dataframe in a .parquet file on disk.

This whole precess takes a few minutes for cm_global_imp_0 and about an hour for pgm_global_imp_0 on a modern macbook.
The second time `dataset.df` is called the cached final dataframe is read from disk and returned in a few seconds.

If you run into memory issues when joining data you can change their definitions in ``views/specs/data/spec.yaml`` to either read a subset of columns, by specifying a list of cols to the cols section of the dataset definition.
Or you can remove one or more tables from the list of tables of the dataset definition in the same file.
All datasets have been successfully joined on computers with 32GB of RAM.


Specification
~~~~~~~~~~~~~~~

Ideally, all operations in views that need source datas should start from a dataset from the ``views.DATASETS`` dictionary.

Managing the huge lists of columns in ViEWS has always been a big problem since the start of the project and has seen many developer hours spent scouring through long files with large column lists in one form or another.
The current preffered solution to this is the ``views/specs/data/spec.yaml`` file.

It defines

* All tables as
    * A name
    * A backing database table
    * A set of identifier columns
* All datasets as
    * A name
    * A reference to a skeleton table
    * A list of references to data tables
    * A set of references to transformations
    * An optional list of columns to subset the sources by.
* All transformations as
    * A name
    * A function reference
    * Parameters to the function

When the user runs ``from views import TABLES, DATASETS`` the ``TABLES`` and ``DATASETS`` dictionaries are defined by parsing this ``spec.yaml`` file.
This parsing is done in ``views.specs/data/__init__.py``
``spec.yaml`` is currently about 3000 lines long, which is as dense as it could possibly be.
The alternative is to keep this information scattered across ``.sql`` files, other specfiles and in python code.


Database update
~~~~~~~~~~~~~~~~

If you are not a member of the ViEWS team you can skip this section.

Each month data must be updated to produce forecasts.

Data updates for a source happen by:

* Fetching the raw source data, timestamping it and storing it on the local machine (not in the database) as a compressed tarfile. This allows quicker development of loaders as data can be fetched once and adjustments made to only the loading part. These may be discarded after loading as we make no effort to be replicable from sources.
* Loading the data from the file on disk and attaching the data source to our identifier system through a skeleton table.
* Interpolating or imputing any missingness.
* Overwriting any existing data currently in the source's schema.
* Pushing imputed and or unimputed tables for the appropriate level of analysis (skeleton) into the source's schema.

Before running a database update, carefully read the code you will be running.
Becuase we don't keep original source data and a load might overwrite the data we have in the database be careful.
Database backups are available but avoiding them is nice.

The interface to update the database is the runner file in ``runners/update_database.py``.
To use it to update ``icgcw`` for example do

.. code-block:: bash

    conda activate views2
    python runners/update_database.py --icgcw

This will fetch the latest ``icgcw`` data from their website, attach it to our id system and recreate the icgcw.cm table.

``update_database.py`` has a --nofetch argument which will stop refetching of the source data.
This is nice when working on a loader as you can fetch once and then load multiple times while you're getting the code correct.

All sources can be updated in parallel, EXCEPT GED and ACLED!
They can not be updated in parallel as they update the same tables.

Adding new sources
~~~~~~~~~~~~~~~~~~~

If you are not a member of the ViEWS team you can skip this section.

First take a look at the existing loaders in ``views.database.sources``.

Then write code to fill the following checklist

* Create a new directory with an ``__init__.py`` in ``views.database.sources`` that exposes a ``fetch_sourcename()`` and ``load_sourcename()`` function.
* Decide on a schema name. If the source is likely to change its list of columns make sure the name has a version indicator in it.
* Write the fetcher. It should write the raw data to a timestamped tar.xz file.
* Write the loader. It should read the timestamped tar.xz file and update or create on or more tables in the source's schema.
* If the source needs imputation the loader should expose both imputed and unimputed tables.
* When ready, add the loader to the ``update_database.py`` runner file and expose it as a command line argument there. Make sure it works by running it through that runner.
* Add entries for the tables produced by the loader to ``views.specs.data.spec.yaml``, both as tables and as ``tables`` members of the datasets.
* Write some documentation for it in the codebook below.

The tricky parts usually come in the form of joining the source to our existing identifiers. Each source is different so take a look at similar sources to get an idea of how to do it.

Good luck!


Model
------

Models are the core building block of the ViEWS system.
A model in ViEWS has

* An outcome, that determines which column to train the model to predict.
* A set of features which control which columns of data the model sees.
* Some estimator. The most commonly used being the RandomForestClassifer from scikit learn.

Models are usually thought of to represent some theme of features, such as conflict history or political institutions or some combination of themes.
By using several models with different themes and studying how their predictions behave we try to understand how they affect the final forecast ensemble, which hopefully lets us make statements like "due to recent violence in X the risk of contiued violence is high in X and surrounding areas.".

ViEWS has gone through many iterations of model specification systems.
The current implementation is defined in ``views/apps/model/api.py`` (API-REFERENCE) as the class Model.
See that file for implementation details.
The fundamental problem the implementation is trying to solve is maintaining and developing a system that allows

* Correct time shifting for producing forecasts.
* Easy specification and evaluation of models for development.
* Ease of automatic use for monthly forecasts.

Combining these two is suprisingly difficult as the sets of models grows large and the extra parameters to the models must be taken into account:

* Features.
* Training data scopes (africa/global, temporal limits).
* Estimator specifications, hyperparameters etc.
* Downsampling.
* Outcomes
* Prediction strategy: step-ahead? dynamic simulation? Onset model? Delta model?
* Step sets.
* Etc.



Example
~~~~~~~~~

Feel free to create a new notebook and paste in the following blocks.
Make sure you have installed the views package and fetched data before you start.
To define a model we create an instance of the Model class.
The following snippet defines a minimal example of a model.

.. code-block:: python

    import json
    from sklearn.ensemble import RandomForestClassifier
    import views
    from views.utils.data import assign_into_df

    # Two periods, A for calibration and B for evaluation of calibrated predictions.
    period_a = views.Period(
        train_start = 121, train_end=396, predict_start=397, predict_end=432
    )
    period_b = views.Period(
        train_start = 121, train_end=432, predict_start=433, predict_end=468
    )

    cols_features = [
        "time_since_ged_dummy_sb",
        "time_since_ged_dummy_ns",
        "time_since_ged_dummy_os"
    ]

    # Month ahead steps to train for
    model = views.Model(
        name = "my_model",
        col_outcome = "ged_dummy_sb",
        cols_features = cols_features,
        steps = [1, 12, 36],
        outcome_type = "prob",
        estimator= RandomForestClassifier()
        periods = [period_a, period_b],
    )


Now to train it.
This will import the Country-month africa data and fit the estimators of the model.
3*2=6 copies of RandomForestClassifier, for each step and for each period, will be fit.
The fit instances of the estimator will be pickled and stored on your local machine in
the structure location format ``views.DIR_STORAGE/models/$modelname_$periodname_$stepnumber.joblib``.
Whenever the model needs the estimators it will read them from disk so you can safely restart your computer and your fitted estimators will be there.
Notice that estimators are stored by model name so if you define two models with the same name their estimators will overwrite eachother.

.. code-block:: python

    # Use the country month dataset for africa
    dataset = views.DATASETS["cm_africa_imp_0"]
    # Get a pandas dataframe of it
    df = dataset.df

    # Fit all the estimators, that is for each period and each step.
    model.fit_estimators(df)


Now to make some predictions.
This will create uncalibrated predictions for the times in the  predict_start - predict_end intervals for both periods and calibrated predictions for the period_b predict interval.
``assign_into_df()`` lets us insert columns into the master dataframe ``df`` in a safe and repeated way.
Assigning in the same column twice but with different temporal coverage will retain values from both periods.
This in contrast to previous setups where we had to maintain separate dataframes for different time periods.

.. code-block:: python

    # Make uncalibrated predictions
    df_pred = model.predict(df=df)
    df = assign_into_df(df_to=df, df_from=df_pred)

    # Make calibrated predictions
    df_pred = model.predict_calibrated(
        period_calib=period_a, period_test=period_b,
    )
    df = assign_into_df(df_to=df, df_from=df_pred)


Now that we have predictions in our ``df`` we can evaluate our model.

.. code-block:: python

    model.evaluate(df)
    # "Pretty" print our evaluation scores
    print(json.dumps(model.evaluation, indent=4))



Step shifting, ss and sc
~~~~~~~~~~~~~~~~~~~~~~~~~

(Sorry about formatting and notation inconsistency in this section, it is TODO)

The core feature that the Model interface provides that is somewhat tricky to do correctly yourself is step shifting predictions.
The Views Model lets user specify which times they want predictions for and for which steps they want estimators fit and solves this.
The Model class time shifts the data for training and prediction and produces the forecasts.
Here is how it works.

Step shifting is one way to produce forecast predictions from models.
We train estimators on the outcome at the present time based on the features ``step`` time periods ago.
We then take the values of our features today and feed them to our trained estimators to produce forecasts ``step`` time periods into the future.
The aim is to predict the outcome (conflict) for all months in between next month and 36 months from now.
So we train estimators for a set of steps in the range 1, 36 and have them produce predictions for the time at ``step`` months into the future.
We then linearly interpolate these step specific (``ss``) forecast predictions from single estimators into step combined (``sc``) predictions.

Consider a model that is to produce forecasts for times 11, 12, 13, 14 and 15.
Data on y (the outcome) and X (the features) is available from time 1 to 10.
The user wants estimators fitted for three steps: 1, 3 and 5.

For step 1 training data is y_t ~ X_t-1.

.. list-table::
    :header-rows: 0

    * - y_10
      - X_9
    * - y_9
      - X_8
    * - y_8
      - X_7
    * - \...
      - \...
    * - y_2
      - X_1

Notice y_1 is not included as no matching feature data is available for X_0.

For step 3 training data is y_t ~ X_t-3


.. list-table::
    :header-rows: 0

    * - y_10
      - X_7
    * - y_9
      - X_6
    * - y_8
      - X_5
    * - \...
      - \...
    * - y_4
      - X_1

And for step 5 training data is

.. list-table::
    :header-rows: 0

    * - y_10
      - X_5
    * - y_9
      - X_4
    * - y_8
      - X_3
    * - y_7
      - X_2
    * - y_6
      - X_1


So when ``model.fit_estimators(df)`` is called, what happens internally is roughly

.. code-block:: python

    for step in steps:
        # Shift features
        df_step = df[cols_features].groupby(level=1).shift(step)
        # Don't shift outcome
        df_step[col_outcome] = df[col_outcome]
        estimators[step].fit(X=df_step[cols_features], y=df_step[col_outcome])

There is more code in the actual implementation that deals with dropping missing values, delta and onset transformations and estimator persistence but this is the core functionality.
``groupby(level=1)`` tells pandas to do the time shifting for each group, group being country or priogrid in our case.
In our example we now have 3 fitted estimators for steps 1, 3 and 5.


Then at prediction time the model will produce 4 columns of predictions, ``ss_1``, ``ss_3``, ``ss_5`` and ``sc`` for the times 11 to 15.

For step 1 the model produces forecasts by taking feature data X_10, x_11, \... X_14 and producing predictions yhat_11, yhat_12, \... yhat_15. Feature data X_t produces forcasts yhat_t+1
For step 3 feature data X_8, X_9, \... X_13 produces predictions yhat_11, yhat_12, \.. yhat_15.
And similarly for step 5. Data from X_6 predicts yhat_11 all the way to X_10 producing foreasts for yhat_15.
These yhat predictions are the step specific or ``ss`` predictions.

To produce our ``sc`` forecast series we match ss predictions by their ``step`` to the prediction window and linearly interpolate them to fill any gaps.
Not that all ``ss`` predictions that are included in the ``sc`` prediction are based on the latest available feature data, X_10.

.. list-table::
    :header-rows: 0

    * - sc_yhat_11
      - ss_1_yhat_11
    * - sc_yhat_12
      - interpolated
    * - sc_yhat_13
      - ss_3_yhat_13
    * - sc_yhat_14
      - interpolated
    * - sc_yhat_15
      - ss_5_yhat_15.


This interpolation is defined in code in ``views.apps.model.api:sc_from_ss`` (REFERENCE).

Two key points here:

* For actual forecasts these ``ss`` values are nonsense for every ``t`` where they are based on values of X from the future (X_11 and onwards) because we don't know the values of X in the future. However, for step 1 yhat_11 is ok as it is based on X_10, for step 3 yhat_13 is ok as it is also created from X_10 and so is yhat_15 from step 5.
* Notice how training data is never used for prediction. For each step the training data X ends exactly one month before the data used to create the predictions.


When calling ``df_pred = model.predict(df)`` The ``sc_yhat`` series in ``df_pred`` has the name from ``model.col_sc``


Calibration and A, B, C
~~~~~~~~~~~~~~~~~~~~~~~~

(Formatting is TODO)

The ``Model`` class also handles calibration.
Calibration is the process of adjusting predicted values from a model to better represent observed values. This is necessary for two main reasons:

* An estimator might produce inaccurate distributions of predictions, most usually because of downsampling in estimator training but it might also be a fundamental limitation in the estimator type. A classifier that discriminates very well might not give a correct probability distribution.
* A model trained on long historical series might produce inflated or deflated predictions compared to one trained on more recent developments. Calibrating to more recent data allows using long series of data for training while retaining recent averages.

Calibration in ViEWS is based on a two step process:

* Produce predictions for a calibration period where actuals are known.
* Produce test predictions to be adjusted.

The algorithm is simple:

* Predict for the calibration period.
* Predict for the testing period.
* Compute calibration parameters based on calibration period predictions and actuals.
* Compute calibrated test period predictions by applying the calibration parameters to the uncalibrated test period predictions.

This happens separately for each step.

The method of obtaining calibration parameters and their application is determined by the model's ``outcome_type`` parameter, which can take two values, either ``"real"`` or ``"prob"``.

For probability value models calibration is done by fitting a logistic regression on ``y_actual ~ beta_0 + beta_1 * logodds(y_pred_calibration) `` and then transforming the test period predictions as  ``y_pred_test_calibrated =  e^(beta_0 + (beta_1 * logodds(y_pred_test))) / (e^(beta_0 + (beta_1 * logodds(y_pred_test))) + 1) ``.

For real value predictions calibration is done by


.. code-block:: python

    # Compute standard deviation ratio
    std_ratio = s_calib_actual.std() / s_calib_pred.std()
    # Remoe the calib mean from test predictions
    s_test_demeaned = s_test_pred - s_calib_pred.mean()
    # Shift calib de-meaned test predictions by the calib actual mean
    # And scale to the std ratio
    s_test_pred_scaled = s_calib_actual.mean() + s_test_demeaned * std_ratio


For implementation see views/apps/model/calibration:calibrate_prob and calibrate_real. (REFERENCE).

The names A, B and C for periods are a common convention in ViEWS that relate to calibration.

* A is the calibration period for B. Prediction period is the three years preceeding B. Currently 2013.01 - 2015.12 (397 - 432). Training period for A is all available data before and including 2012.12 (396)
* B is the evaluation partition where calibrated constituent models and ensemble performance are evaluated. B is also the calibration period for C. Prediction period for C is the three latest years with yearly release data from the UCDP. Currently 2016.01 - 2018.12 (433-468). Training period for B is all avaialble data before and including 2015.12 (432)
* C The forecast partition, which is the month after the latest available monthly update UCDP GED data and 37 months forward. This is a rolling window updated each month. Training is all the data before and including 2019.12 (480)

The training periods for A and B are all data up to the month before their first

To get the time limits of these periods in python for a given ``run_id`` do

.. code-block:: python

    from views.specs.periods import get_periods_by_name
    periods_dict = get_periods_by_name(run_id="r_2020_06_01")
    print(periods_dict)

Delta models
~~~~~~~~~~~~~

A special case of model is the delta model.
The goal of a delta model is to predict a change from the current time.
Because models train estimators for each step ahead this delta transformation must be done for each step ahead.
To define a model that automatically delta transforms the outcome for each step in estimator fitting pass in the ``delta_outcome=True`` keyword argument to the Model constructor:

.. code-block:: python

    import views
    from sklearn.ensemble import RandomForestRegressor

    # Month ahead steps to train for
    model = views.Model(
        name = "my_model",
        col_outcome = "ln_ged_best_sb",
        cols_features = cols_features,
        steps = [1, 12, 36],
        outcome_type = "prob",
        estimator= RandomForestRegressor()
        periods = [period_a, period_b],
        delta_outcome=True, # <-- This argument
    )

When this is done the outcome column in the training data is delta transformed separately for each step before fitting the estimators.

Intuitively it can be helpful to think of ``ss`` predictions from delta models to be predicted change over ``step`` months predictions.

Note that the outcome column, in the example above ``ln_ged_best_sb`` should not have any delta transformation applied before, the model object does the transformation.


Onset models
~~~~~~~~~~~~~

Another special case are onset models.
An onset is defined as the first occurence of an event in a given time window.
So a country in conflict months 4, 6, and 10 would have onsets at 4 and 10 if the onset window was defined as 3 months.
Onset models need special treatment from the Model object as they should only be trained on rows where an onset is possible. Meaning countries or grids that are in a state of ongoing conflict are dropped from the training data.
To define an onsest model:

.. code-block:: python

    import views
    from sklearn.ensemble import RandomForestClassifier

    # Month ahead steps to train for
    model = views.Model(
        name = "my_model",
        col_outcome = "ged_dummy_sb",
        cols_features = cols_features,
        steps = [1, 12, 36],
        outcome_type = "prob",
        estimator= RandomForestClassifier()
        periods = [period_a, period_b],
        onset_outcome=True, # <-- This argument
        onset_window=12, # <--- And this argument
    )


Currently onset models only affect training, predictions are not restricted. So an onset model may still predict an onset the first month after an ongoing conflict. This is being worked on.



Ensemble
----------

Ensembles are another core part of ViEWS.
Ensembles represent collections of models.
They work by combining predictions from the constituent models, hopefully giving a more accurte prediciton than any single model.

To define an ensemble use the ``Ensemble`` class:

.. code-block:: python

    from views import Ensemble, Model, Period


    # See previous examples
    model_a = views.Model(...)
    model_b = views.Model(...)
    models = [model_a, model_b]
    period_calib = Period(...)
    period_test = Period(...)
    periods=[period_calib, period_test]
    ensemble = views.Ensemble(
        name="my_ensemble",
        models = models,
        outcome_type="prob",
        col_outcome="ged_dummy_sb",
        method="average",
        periods=periods,

    )


To get predictions from an ``Ensemble`` you must first compute predictions for the constituent models.

.. code-block:: python


    # Compute constituent model predictions
    for model in models:
        df_pred = model.predict(df)
        df = assign_into_df(df, df_pred)
        df_pred = model.predict_calibrated(
            df=df,
            period_calib=period_calib,
            period_test=period_test
        )
        df = assign_into_df(df, df_pred)

    df_pred = ensemble.predict(
        df=df,
        period_calib=period,
        period_test=period,
    )

``df_pred`` now contain simple unweighted average predictions from the constituent models for ``period_test.times_predict``.
Column names are in ``ensemble.col_sc`` and ``ensemble.cols_ss``.

And to evaluate you simply do

.. code-block:: python

    ensemble.evaluate(df)
    print(json.dumps(ensemble.scores, indent=4))



EBMA
~~~~~
Ensemble Bayesian Model Average, or EBMA, is the second method.
It is implemented using the EBMAForecast package from R.
In essence, EBMA calibrates constituent model predictions and learns weights to apply to them.
Because EBMAForecast is an R package, and ViEWS is primarily a python project, the implementation is a bit of a hack. It:

* Writes data to .csv files in temporary directories.
* Parametrises a boilerplate Rscript file with locations to that data.
* Calls R in a subprocess with the filled Rscript that writes results to another .csv file.
* Returns that data and model weights.

For details see ``views/apps/ensemble/ebma.py``.
Users wishing to use the EBMA functionality must install R and the EBMAforecast package themselves.
Unfortunately, EBMAforecast was recently dropped from CRAN, the R package repository.
It is still available from the archvies though: For a script that installs EBMA see views/apps/ensemble/templates/install_ebma.R.

When the ``Rscript`` program is available on your shell and has the EBMAforecast package installed you can go ahead and use it with the exact same syntax as a regular unweighted average ensemble. Just change the ``method`` parameter from "average" to "ebma".

One very useful feature of EBMA are the model weights, that tell you how EBMA valued each model.
After running ``ensemble.predict(...)`` on an EBMA ensemble the model weights are available for inspection through the ``ensemble.weights`` dictionary.



Dynasim
-------

Dynasim is a dynamic simulation program that is currently being integrated into this repository.

There are many implementations but the one ViEWS is currently using to produce forecasts can be seen at https://github.com/UppsalaConflictDataProgram/OpenViEWS/tree/master/ds .
That implementation was the author's (mine) first large python project and its quite difficult to interface with.
A rewrite is in progress and is ironing out some final bugs.
It will be added to this repository soon.


Specification
--------------


Models and Ensembles
~~~~~~~~~~~~~~~~~~~~
Specifying models and ensembles has long been a pain point.
One one hand, maintaining long lists of columns and other options in python with various list comprehensions and filters becomes very difficult to understand and document as the number of lines of code of just definitions approaches the thousands.
On the other hand a pure specification file approach in YAML is inflexible and difficult to parse.
I think we are now approaching a decent hybrid solution.

The definition of models and ensemble now happen as instances of the ``Model`` and ``Ensemble`` classes.
While model development frequently happens in notebooks and miscellaneous scripts the models that go into the production pipeline live in ``views/apps/pipeline/{models, ensembles}_{cm, pgm}.py`` .
Any finalised models deemed ready for prime-time should go in there or a similar file where they are importable from a script in the ``runners`` directory.


Models are defined as instances of the ``Model`` object in plain python.
Arbitrarily complex estimator instances, step sets, training period restrictions or tags can be defined there.

To aid in specification a set of YAML files and a parser exists in ``views/specs/models`` They expose a set of dictionaries. To get the list of columns for model ``sb_allthemes`` you can do

.. code-block:: python

    from views.specs.models import pgm
    print(pgm["sb_allthemes"]["cols_features"])

The files in ``specs/models/{am, cm, pgm}.yaml`` make use of column sets to organise columns and themes to organise column sets before finally combining themes and an outcome column to define the columns for a model.
The aim of this is to reduce errors in model specification and make working with large sets of features easier.

Ensembles are defined as lists of models in ``views/apps/pipeline/ensembles_{cm, pgm}.py``

Both models and ensembles can be imported and used by runners, or any other scripts, like so:

.. code-block:: python

    from views.apps.pipeline import models_cm
    model = models_cm.all_cm_models_by_name["cm_sb_reign_global]

    model.fit_estimators(df)
    model.predict(df)

If cached estimators for the model are already on your system they can be reused and .fit_estimators(df) can be omitted.

Periods
~~~~~~~~

Keeping track of ``month_id`` can be tricky.
To aid in this standard time limits are defined as ``Period`` objects in ``views/specs/periods.yaml``. They can be imported and used like so:

.. code-block:: python

    from views.specs.periods import get_periods_by_name

    # Can now be used to make predictions, calibrate or define training times.
    period_a: views.Period = get_periods_by_name(run_id="r_2020_06_01")["A"]
    period_b = get_periods_by_name(run_id="r_2020_06_01")["B"]



Plots
------

TODO

Evaluation
~~~~~~~~~~

TODO

Maps
~~~~~

TODO


Run
----

An important part of the effort to rewrite a lot of ViEWS code was the difficulty in doing a complete run, from data ingestion to published forecast.
It currently involves many steps of manual intervention from at least three people, which is fragile and tedious and frequently gets delayed.

This rewrite provides a single common interface for all the tasks that go into a run which is the ``runners`` directory at the root of the repository.

The outline of the process will soon be as follows:

* Fetch and load monthly data to database. (Done, ``runners/update_database.py``)
* Refresh locally cached data. (Done, ``runners/refresh_data.py``)
* Compute dynasim predictions. (TODO)
* Compute step-ahead constituent model predictions. (Done, ``runners/predict.py``)
* Compute ensemble (Done, ``runners/predict.py``)
* Produce plots (TODO)
* Produce report (TODO)
* Publish (TODO)

Some of these parts will require manual work of course, especially some interactions with our compute cluster and report preparation and publication.
The goal is to automate what can be automated as much as possible.



Tooling
========


git
~~~~

TODO

run\_tools.sh
~~~~~~~~~~~~~

TODO

docs
~~~~~

TODO

Text editors / IDE
~~~~~~~~~~~~~~~~~~~

TODO


Codebook
=========

We are working on a comprehensive and complete codebook of all ViEWS columns.
Sorry for the current state of this incomplete data documentation, this is a work in progress.

Views data sources
-------------------

All columns used in views are prefixed with a source specific acronym.
While we sort out the final codebook see notes below for each source.


ACLED ``(acled_)``
~~~~~~~~~~~~~~~~~~~~

ACLED is the armed conflict location event data.
ViEWS recodes ACLED into approximations of UCDP GED categories of violence.
There are thus 8 primary columns exposed by ACLED in ViEWS data.

* acled_count_pr: Protest event count
* acled_count_sb: State-based violence event count
* acled_count_ns: Non-state violence event count
* acled_count_os: One sided violence event count.
* acled_fat_pr: Protest fatality count
* acled_fat_sb: State-based violence fatality count
* acled_fat_ns: Non-state violence fatality count
* acled_fat_os: One sided violence fatality count.

``acled_dummy_[pr, sb, ns, os]`` are dummy encodings of ``acled_count_``.

FVP ``(fvp_)``
~~~~~~~~~~~~~~~
A country year dataset compiled for a another project. Combining data from VDEM, WDI, EPR.

* Columns prefixed ``prop_`` are from EPR, see https://icr.ethz.ch/data/epr/
* Columns prefix ``ssp2`` are from SSP, see https://tntcat.iiasa.ac.at/SspDb/
* auto, demo, electoral, etc are from VDEM, see https://www.v-dem.net/en/

Comprehenseive documentation is TODO.

GED ``(ged_)``
~~~~~~~~~~~~~~~
The main outcome of ViEWS comes from UCDP - GED.
See https://ucdp.uu.se/downloads/ for details.

6 main columns are exposed from GED:

* ged_best_sb: Best estimate of fatalities for state-based violence.
* ged_best_ns: Best estimate of fatalities for non-state violence.
* ged_best_os: Best estimate of fatalities for one-sided violence
* ged_count_sb: Number of events for state-based violence.
* ged_count_ns: Number of events for non-state violence.
* ged_count_os: Number of events for one-sided violence.

With the transform ``ged_dummy_[sb, ns, os]`` dummy encoding ``ged_count_[sb, ns, os]``


ICGCW ``(icgcw)``
~~~~~~~~~~~~~~~~~~~

The international crisis group has an online conflict tracker at https://www.crisisgroup.org/crisiswatch

This is scraped and updates are encoded in 5 columns:

* icgcw_alerts: Appeared in an alert
* icgcw_deteriorated: Situation deteriorated
* icgcw_improved: Situation improved
* icgcw_opportunities: Opportunity spotted.
* icgcw_unobserved: Country doesn't appear.



PRIOGRID ``(pgdata_)``
~~~~~~~~~~~~~~~~~~~~~~~~

Priogrid data is fetched from the PRIO-GRID API at https://grid.prio.org/#/apidocs.
For full codebook see https://grid.prio.org/#/codebook
41 columns are exposed from priogrid with their original names retained.
Columns where an yearly (_y) and an static (_s) version are sometimes taken the MAX() of to combine them.


REIGN ``(reign_)``
~~~~~~~~~~~~~~~~~~~~

REIGN Rulers, Elections, and Irregular Governance dataset.
For details see https://oefdatascience.github.io/REIGN.github.io/

SPEI ``(spei_)``
~~~~~~~~~~~~~~~~~~~~
SPEI GLobal Drought monitor.
For details see https://spei.csic.es/map/maps.html


VDEM ``(vdem_)``
~~~~~~~~~~~~~~~~~~~

Varieties of democracy. Version 10 is currently loaded.
For codebook see: https://www.v-dem.net/en/data/data-version-10/
Columns loaded from the ``Country-Year:V-DemFull+Others`` file.
Columns ending in the following suffixes are currently not included due to memory constraints.

* _codehigh
* _codelow
* _ord
* _sd
* _mean
* _nr
* _osp



WDI ``(wdi_)``
~~~~~~~~~~~~~~~~
World Bank World Development Indicators.
Updated as of May 2020.
Downloaded from http://databank.worldbank.org/data/download/WDI_csv.zip
For details see https://databank.worldbank.org/source/world-development-indicators

Transforms
~~~~~~~~~~~

ViEWS has many transformations.
The naming convention is that the transform name and parameters are prepended to the column name:
``transform_a(transform_b(col, params_b), params_a)`` is named ``transform_a_params_a_transform_b_params_b_col``

* delta_col: Time delta: col - tlag_1(col)
* greq_value_col: Greater or equal dummy encoder.
* smeq_value_col: Smaller or equal dummy encoder
* in_range_low_high_col: Dummy encoder for
* tlag_time_col: Time lag.
* tlead: Time lead.
* ma_time_col: Moving average over time
* cweq_value_col: Count while col equals value.
* time_since_col: Time since column != 0. Implemented as time-lag of 1 of count while col equals 0.
* decay_halflife_col: Exponential decay function
* mean_col: Time-invariant mean of col
* ln_col: Natural log of col
* demean_col: De-meaned values of col. Is col - mean(col).
* rollmax_window_col: Rolling max of time window
* onset_possible_col: Onset possible if no event occured in the preceeding `window` times
* onset_window_col: Onset is 1 if onset is possible and an event occured. 1 for first event in time window.
* sum_cols: Sum of columns
* product: Product of columns
* spdist_col: Spatial distance to closest cell or country where col == 1.
* stdist_k_tscale_col: Space-time distance to closest k cells or countries where col == 1.
* splag_first_last_col: Spatial lag. Sum of col for all neighboring geographic units from first to last order neighbor. So splag_1_1_ged_dummy_sb is sum of ged_dummy_sb in immediate neighbors. splag_1_2_ged_dummy_sb is the sum of ged_dummy_sb in neighboring geographies and their neighbors. splag_2_2 would give a hollow circle of just neighbors neighbors, but not direct neighbors.


Planned codebook
-----------------

We are working towards building a complete codebook dictionary that for each source describes each column in the following structure:

.. code-block:: yaml

    col_one:
        short: "A short description of col_one"
        long: "A very long description of col_one"
    col_two:
        short: "Short desc of col_two"
        long: "Very long desc of col_two"

That can then be parsed into a nice looking complete codebook.
Some sources facilitate this by providing a csv describing each column that can be automatically parsed.
Others are more tricky and take more work.
Until that is done (there are just shy of 10 000 columns in total), please see the original source's codebooks.


