# Misc?

## Static dependencies list

To avoid issues with breaking changes from updated dependencies the
main installer now uses a static list of versioned dependencies in env_static.yaml.
No more updated dependencies suddenly breaking code.

If you want to add a dependency:

* add it to environment.yaml in this dir,
* recreate the views2 environment with (from this dir):

    conda remove --name views2 --all --yes
    conda env create -f environment.yaml

Then run

    ./freeze_env.sh

to update env_static.yaml.
