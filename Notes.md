Notes about the code and different files in it

# contrastive_training.py
- It first initializes arguments and sets the script to use one GPU
- Runs init experiment, which is responsible for creating the paths for logging and checkpoint saving. It also creates a Tensorboard writer and gives it some of the arguments.
- Then it loads the VIT Dino pre-trained model, and freezes all layers except some of the last blocks (default from 11/11)
- Then it creates the augmentation transformer for the train and test data, and it creates an augmenter to generate multiple views of the training data for contrastive learning
- Then it loads the datasets with these transforms, and creates a sampler to balance labelled and unlabelled classes. The sampler assigns 1 weight to labeled data and len(all_data)/len(unlabeled_data) weight to unlabeled data
- Then the dataloaders are made and the prediction head is created. I have no idea why the prediction head is made alone, and I should **invistigate that further**

## Training
- Pretty much same as in SimGCD, just without mixed precision and using K-Means instead of parametric classification. No clustering training is done in this script
- Unlike in SimGCD it supports running unsupervised contrastive loss only on unlabelled data, while SimGCD can only do it with all data
- SimGCD normalizes projections only for supervised loss, while this one does it for both
- Does accuracy calculation every batch, while SimGCD does it every epoch

## Testing
- Goes through test set and gets predictions for all images. Also creates a labels array and a mask array, which designates each prediction/value to open or closed set
- Fits all data using sklearn's KMeans algorith. Number of clusters is equal to number of classes
- Does not really do any supervised KMeans

# estimate_k.py
- Uses a features dataset that takes an original dataset, then replaces the images by the pre-computed features. It loads features on request and not all in the start. Will result in overhead if passing through the dataset more than once.
- Estimates optimal number of classes from the training dataset
- Supports two methods for estimating K:
  - Using an inbuilt scipy optimzer based on Brent's algorithm
  - Using binary search
- Both utilize a test_kmeans function that uses sklearn's inbuil K-Means method. This also does not do any kind of supervised K-Means. Both test_kmeans functions are essentially identical. They just return opposite sign values.
- There are multiple score functions, such as cluster, normalized mutual info (nmi) and adjusted random (ari) score. I'll read about them later
- Search algorithms are only concerned with labelled clustering accuracy

# k_means.py
- Creates dataset and target transform just like in contrastive learning, than converts to features dataset
- Has an ominous "TODO: Debug" comment
- Mask_cls is a mask for open set, while mas_lab is a mask for unlabelled data. **I kinda still do not get the difference**

# K_Means Class
- Has a main function for testing, which can be useful.
- Fitting function tries to fit K-Means n_init times, and returns the values from the best K-Means fit
- Each time a K-Means fit is to be run it run the fit_once function with up to max_iterations
- Fit once first assigns the labeled centers as the mean of their corresponding labeled features, then it uses a modified K-Means++ algorithm to initialize the rest of the centers
- During iterations it find the distance between unlabelled features and all centers, while distance of labelled features are only computed with labelled centers
- I will need to **refresh my knowledge on K-Means**, but there seems to only be two distance functions


# Notes
- There is no script that actually tests the computer model and clusters on the test set

# Development
- Understand the contrastive loss and adapt it to hyperbolic space (Look at Meru for reference)
- Currently uses Scikit's K-Means algorithm. Need to adapt the semi-supervised K-Means algorithm in the code to work in hyperbolic space and work with fully supervised data.
- All functions are available in MERU, so I just need to copy them and adapt the model to work with hyperbolic mapping
