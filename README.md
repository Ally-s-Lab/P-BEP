# Probability-Based Editing Predictor (P-BEP)
A meta-predictor for adenosine to inosine (A-to-I) editing sites based on base-pairing probabilities.  

## Getting started

In order to convert genomic positions to vectors of probabilities you will need to install ViennaRNA on your system 
(make sure you have sudo privileges).  
##### Installing ViennaRNA
- Use your distribution's package manager to install cmake
- Visit the ViennaRNA homepage and follow the instructions to install ViennaRNA to the project main directory
    - Alternatively, execute `bash install_viennaRNA.sh`  
##### Downloading GRCH38 chromosomal DNA files
In order to convert many genomic sites to probability vectors in a reasonable time, you will need to download the primary assembly.  
 - Change directory to src
 - Modify download_grch38:
    - Enter your email in the line `Entrez.email = ""` between the quotation marks
 - `python3 -m download_grch38` 

A list of chromosome sequences will be downloaded and saved to lib/sequences.pkl as a pickle file. 
 
You can now use a pre-trained model to classify sites or train a model from scratch.
## Using and training models

To predict editing sites using processed true positives and true negatives files, such as
lib/testing_positives.csv and lib/testing_negatives.csv, use the ensemble model module:
`python3 -m ensemble_voting_classifier`.

Using only pre-processed data supplied in this repository will not require any additional data processing.

### Generating base-pairing probabilities

Whenever you wish to use new genomic sites for training, testing or prediction, you will need to process them first.
Note that you should execute download_grch38.py first (see getting started).

The models work directly with features, so genomic locations will need to be translated to features.
The format of the training file needs to be comma seperated values (CSV) of 51 probabilities between 0 and 1, followed 
by a 0 (False) or 1 (True) classification.  

To create the vector of probabilities for a specific position, use the function "coord_to_prob(chromosome, position)" 
from the module "coord_to_prob.py". The result is a numpy 1d vector of length 51, containing the probability to be 
base-paired for the 25 downstream and upstream bases, including the input base itself. These vectors can be used as 
samples for working with models directly. The input you use should be a
string or integer (23 for X and 24 for Y) for the chromosome, and an integer for the position.
Examine the file lib/training1.csv for a processed data file example.  

The file lib/empty_training_file.csv is a servicable but empty file (with a header) you can populate and use.
Finally, you may need to modify the following line, based on the version of ViennaRNA that was downloaded:  
`os.system("""echo "{}" | ../ViennaRNA-2.4.15/data/usr/bin/RNAplfold -u 1 -o""".format(sequence))`

### Using a pre-trained model

The true ensemble classifier is a majority-vote among 3 different models. To use the classifier,
review the config.yaml file and verify the data you wish to use. Then simply execute
`python3 -m ensemble_voting_classifier`. # SAVE RESULTS TO SOMEPLACE

### Training a model

##### Training base models
In each of the modules "CNN.py", "FCNN.py", XGBoost_train_load_and_pred.py":  
 - Verify the data you wish to use is entered correctly in config.yaml
 - Execute the module - e.g `python3 -m CNN`

##### Training the metapredictor

 - Verify the models you wish to use are entered correcly in config.yaml
 - Execute the module - `python3 -m ensemble_voting_classifier`
 
 ### Making predictions
 
...


