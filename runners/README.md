# Runners

Runners are entrypoint scripts to ViEWS functionality for

* Training
* Predicting
* Evaluating

They should be as simple as possible, with complexity handled in the apps themselves.
Entrypoints should only handle

* Dealing with execution context (slurm, conda etc).
* Parsing arguments
* Logging
* Executing the correct functionality from modules

