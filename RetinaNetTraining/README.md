## RetinaNet Training
This directory contains the files necessary to train a retinanet_resnet50_fpn_v2 RetinaNet
model using the resnet50_fpn as the backbone. The files are set up assuming that k-fold 
cross-validation is being used (i.e. there is labeled validation data available).
The following files are available:

### 1. `training_retinanet.py`

The main script called from the terminal. Sets up the transformers, datasets, dataloaders, custom
model, runs training with validation checks, and finally shows the final model's validation results.
The outputs of the final validation run assume at most 2 object classes detected: 
1. Prostate alone.
2. Prostate and Bladder.

If different data is given to the model, the labelling of predictions will need to be updated.

### 2. `CustomRetinaNet.py`
Custom RetinaNet class used by `training_retinanet.py`. The transformers are also included as
functions, with a training transformer and a validation transformer given. The class inherits
from `RetinaNet` and only makes a change to the forward function. The forward function can return
calculated losses if required. This is used under the assumption of k-fold cross-validation
where the validation data is labelled. If only detection are required (i.e., errors cannot be
calculated or are not wanted) the `return_losses` parameter can be set to `False`.

It is assumed that a new model is being trained, i.e. no fine-tuning of previously trained models.

### 3. `EarlyStopping.py`
This class monitors the validation errors and if there has not been an improvement over a 
certain number of epochs (patience) further training is cancelled. The improvement limit
is controlled with `delta` (or `patience_delta` in the terminal call). If an improvement 
is detected the best model, and only the best model, is saved. If training is interrupted
the last saved model may not necessarily be the latest model.

### 4. `ProstateDataset.py`
Custom Dataset class for abdominal ultrasound scans of the prostate where the prostate and bladder
bounding boxes are available. Assumes bounding boxes are given as normalised 
(x_centre, y_centre, width, height) in the labels files and changes them to (x_min, y_min, x_max, 
y_max) in pixel coordinates. Minor dataset validation is done to ensure there are the same
number of labels as images, and that the same naming convention is used between images and labels.
If a different dataset is used, this will likely need to be updated.

### 5. `utils.py`
Utils functions:
1. Plot losses as training is taking place.
2. Generate the arg parser.
3. Generate validation images with detected boxes.