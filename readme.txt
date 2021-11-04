This folder contains the codes to reproduce the analysis and figures from Olinger et al 2021.

The entire project is contained within the notebook "riftquake_analysis.ipynb."

Data downloaded from IRIS will be placed in data/MSEED/raw and data/XML. Appropriate directories for each station and component will be made in data/MSEED/no_IR when the instrument response is removed from the raw data.

Individual code files are located in the folder that corresponds to the code type. For instance, the files to prepare templates and run template matching are in detection/template_match.py.

Outputs for each step are dumped into the respective folder within the directory outputs. For instance, figures will be placed in outputs/figures.

-S
