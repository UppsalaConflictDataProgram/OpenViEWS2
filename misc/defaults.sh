# This file contains default values, don't change it.
# Make your changes to sourceme.sh in the root of the repository.

export VIEWS2_DB_CONNECTSTRING="postgresql://USERNAME:password@HOSTNAME:5432/DATABASE"
export VIEWS2_DB_SSL="False"
export VIEWS2_DB_SSL_CERT="~/postgres/postgresql.crt"
export VIEWS2_DB_SSL_KEY="~/postgres/postgresql.key"
export VIEWS2_DB_SSL_ROOTCERT="~/postgres/root.crt"

export VIEWS_DIR_STORAGE="" # Emtpy string will default to the storage directory in the repo