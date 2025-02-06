import numpy as np
from preprocess import train_dataset

class_labels = np.array(train_dataset.class_names)
print("Class Labels:", class_labels)
