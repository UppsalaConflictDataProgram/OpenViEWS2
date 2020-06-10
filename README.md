# ViEWS2

Getting started

Download and install miniconda3: https://docs.conda.io/en/latest/miniconda.html
After you have conda installed, in your terminal run

    ./install_views2.sh

This will create a conda environment called views2 and install the views package there.
To fetch the latest public data run

    conda activate views2
    python runners/import_data.py --fetch

To start using ViEWS code simply run

    conda activate views2
    jupyter notebook

A web browser should open with the jupyter notebook browser.
If you wish to take part in the prediction competition, see projects/prediction_competition/
An example notebook to get you started modelling is in projects/model_development/examply.ipynb.

We develop ViEWS on Mac and Linux computers, the procedure is slightly different for Windows and we haven't developed a streamlined process for it yet.

To open the HTML documentation from here on MacOS run

    ./run_tools.sh
    open docs/_build/html/index.html

And it will take you to the locally built html documementation in your default browser.

To view .pdf documentation (a work in progress) see https://views.pcr.uu.se/download/docs/views.pdf