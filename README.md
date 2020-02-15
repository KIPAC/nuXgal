


This is an analysis code for studying cross corellations between neturino events and galaxy samples.

Some test codes are in the 'tests' sub-directory.
The following steps install the code. 

#### Clone the repo from github
git clone https://github.com/KIPAC/nuXgal.git

#### If you want the tag corresponding to the paper draft you can checkout that version explictly
git checkout tags/v0.1

#### We recommed that you explictly tell the code to put IRFs and ancillary files with the code itself.  You can do this by setting:
export NUXGAL_DIR=DIRECTORY_WITH_CODE

#### Initiate a conda enviroment 
conda create -n nuXgal python=3.6 

conda activate nuXgal

#### Setup the code
python setup.py develop

#### Download IceCube data, for example the public three-year point source data (https://icecube.wisc.edu/science/data/PS-3years) to directory $ICECUBE_DATA_FOLDER.

#### Generate instrument response function
python scripts/generateICIRFS.py -i $ICECUBE_DATA_FOLDER

#### To produce the figures in the paper
python scripts/figures.py 
