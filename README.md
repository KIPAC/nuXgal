# nuXgal
Analysis code for studying cross corellations between neturino events and galaxy samples.

Some test codes are in the 'tests' sub-directory.



# to setup the code environment 
python setup.py 

# download IceCube public three-year point source data (https://icecube.wisc.edu/science/data/PS-3years) to directory $ICECUBE_DATA_FOLDER
# generate instrument response function
python scripts/generateICIRFS.py -i $ICECUBE_DATA_FOLDER


# To produce the figures in the paper 
python scripts/figures.py 
