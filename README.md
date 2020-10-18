# editing-site-pred
A meta-predictor for adenosine to inosine (A-to-I) editing sites based on base-pairing probabilities.  

## Getting started

In order to convert genomic positions to vectors of probabilities you will need to install ViennaRNA on your system (make sure you habe sudo privileges).  
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

### Using a pre-trained model

Prepare your sites in a tab seperated values (TSV) format with no header in the following format:  
Chromosome  Position  
For example:  
1&nbsp; &nbsp; &nbsp; 149121  
22&nbsp; &nbsp; 11311213  
X&nbsp; &nbsp; &nbsp; 14312

### Training a model

If you wish to use your own training data, you will need to prepare a training file manually.
##### Generating base-pairing probabilities

The format of the training file needs to be comma seperated values (CSV) of 51 probabilities between 0 and 1, followed 
by a 0 (False) or 1 (True) classification.  
To create the vector of probabilities for a specific position, use the function "coord_to_prob(chromosome, position)" 
from the module "coord_to_prob.py". The result is a numpy 1d vector of length 51, containing the probability to be 
base-paired for the 25 downstream and upstream bases, including the input base itself.  
Examine the file lib/training1.csv for a training file example.  
The file lib/empty_training_file.csv is a servicable but empty file (with a header) you can populate and use.
Finally, you may need to modify the following line, based on the version of ViennaRNA that was downloaded:  
`os.system("""echo "{}" | ../ViennaRNA-2.4.15/data/usr/bin/RNAplfold -u 1 -o""".format(sequence))`
##### Training models
In each of the modules "CNN.py", "FCNN.py", XGBoost_train_load_and_pred.py":  
 - Edit the line `df = pd.read_csv('../lib/training.csv')` with the path of your training file, relative to src/.  
 - Verify your working directory is src
 - Execute the module - e.g `python3 -m CNN`

Training the metapredictor - 