# mint-analysis
Collection of scripts for MINT tests


## Installation

Clone the repo

To install packages I recommend using uv [https://docs.astral.sh/uv/], which
is a fast modern python package handler that also handles virtual environments.

You'll need to install this first which is shown here [https://docs.astral.sh/uv/getting-started/installation/].

Once you have it installed go to the cloned repo and inside you should see a pyproject.toml file
this contains all the info on the packages needed.

To install the packages simply run `uv sync`and they should all install.

Then to open a jupyter notebook: `uv run --with jupyter jupyter notebook`. (In fact you don't even need the `uv sync` and can
simply do the run, uv will handle the rest). This will run a jupyter notebook instance inside the virtual environment.

Some packages may cause issues such as h5py which may need to be installed separately via brew.

All new instances you make are completely independent so can use different package/python versions as
specified in the pyproject.toml file.
