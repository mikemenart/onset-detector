Dataset is downloaded from http://www.tsi.telecom-paristech.fr/aao/en/2011/07/13/sound-onset-labellizer/ and 
put into a directory named Leveau. The MACOS folders were removed from the GoodLabels and LabelsPL directories
to simplify the code. 

Leveau.py is responsible to for reading in a processing the dataset
Onset.py uses Leveau.py to access the dataset, and creates the training data on the fly. 
It then trians the network. 