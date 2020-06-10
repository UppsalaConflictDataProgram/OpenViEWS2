# Documentation

To build the documentation cd to this directory and run

`sphinx-apidoc -o source/ ../views`
`make html`

Or just run the run_tools.sh script in the root of the repo. It does this for you.

Human written source files should go in human_source.
Leave the `source` directory to sphinx-apidoc so that we can delete and rebuild it should it break.

## PDF
To build a pdf make sure you have latexpdf installed (miktex worked for me) and run

    make latexpdf

You will get a views.pdf in `_build/latex/views.pdf`.
