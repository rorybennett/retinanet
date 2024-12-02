"""
Results from 5-fold cross validation on the prospective datasets. Only looks at the combined models.

Each model is loaded then its validation images are passed through. Only the top detections are considered (highest
confidence bladder and prostate). Performance metrics are calculated and results plot to be saved.
"""
import RetinaNetTraining.CustomRetinaNet
import RetinaNetTraining.ProstateDataset