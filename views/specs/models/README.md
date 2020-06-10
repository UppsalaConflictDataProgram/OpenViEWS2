# Model specs

ViEWS has a lot of big models with very many features.
Keeping track of which column goes where can be very difficult.
This module provides three master spec files, am.yaml, cm.yaml and pgm.yaml
to attempt to keep track of them.

The idea is to group columns into a hierarchy of
* colsets, that list plain columns
* themes, that groups colsets and other themes
* formulas, that resolve a list of columns from the above

Colsets, or column sets, are simply lists of columns with a name.
Themes are made of colsets or by combining themes and colsets.
Finally, formulas map all columns from a theme or colset to an outcome column.
By applying the solver to these spec files we get solved formulas.
They have a name, a col_outcome and a list of cols_features, which is found by recursively looking them up through themes and colsets.
For a minimal example see tests/test_specs.py

