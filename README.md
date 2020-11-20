### The application of Unet for seismic images segmentation task.

Unet implementation is from https://github.com/jakeret/unet with slight modifications.

#### The entire process is run from the main.py file, and all the main settings are defined there. The detailed description of parameters is given in the comments in the file itself.

To run a model with a particular set of parameters set **n_ samples** to **1** make all hyperparameters to be lists containing a single value (**layer_depth_prior** = **[3]**).  
Training outputs are the following: txt values with accuracy and runtimes, folders for each model created with a particular hyperparameter set containing hyperparameters file, txt files with loss and accuracy history values, corresponding plots; probability segy cubes and a prediction segy

**main.py** – both training and prediction processes are run from this file, all the major settings are defined there  
**train_wrapper.py** – contains a function that runs a training loop and saves results  
**utils.py** – contains miscellaneous functions to save results  
**unet.py** – contains the model implementation  
**predict.py** – contains a function that runs prediction and save results  
**data_loader.py** – contains function to read and preprocess input data  
**schedulers.py** – contains learning rate schedulers implementation from the original Unet repository  


