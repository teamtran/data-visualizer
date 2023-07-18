# Standard Operating Procedure for using spectrometry-visualizer from ZERO
## IF YOU HAVE NOT DONE SOP_github.md, please do it before proceeding.

## Install anaconda
### MacOS
0. Go to https://docs.conda.io/projects/continuumio-conda/en/latest/user-guide/install/macos.html
1. Follow the instructions on the website. Steps 1-6.

### Windows
0. Go to https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html
1. Follow the instructions on the website. Steps 1-5.

## Create new environment for spectrometry-visualizer with the necessary packages
1. In terminal, write this command: `conda create --name spectrometry-visualizer --file requirements.txt` (Keep in mind that you must be in the same directory as the file "requirements.txt")
2. Type in "y" and enter.
3. Next, write this command: `conda activate spectrometry-visualizer`
4. Sanity Check: make sure `(spectrometry-visualizer)` is at the beginning of the line in your terminal.

## Change the path of the directory with all of your data.
### It's incredibly important to be organized and keep your data with an organized folder structure
1. Go to `data` > `data_utils.py`.
2. Change the variable `data_path` to the correct directory where your data is located. Refer to the existing code as a reference to your path.
3. 