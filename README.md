# Coffeapp 

## Important custom functions implemented throghout the research

1. **classes_to_binary**: Takes the input folder from_folder, converts the class labels from multiclass (5 classes: green, green-yellow, cherry, raisin and dry) to binary (ripe and unripe) format using the mapping in the map dictionary, and saves the converted labels in the output folder to_folder.
2. **count_objects**: Counts the number of objects (coffe fruits) in the training and validation sets by iterating through the label files and counting the number of lines.
3. **classes_to_mono**: Similar to classes_to_binary, but converts the class labels to monochrome format (i.e., 0 for all classes, coffe fruits).
4. **split_data**: Splits the input folder into training and validation sets. The percentage of data to use for training is set by the split_percentage argument (default is 0.8).
5. **count_imgs_labels**: Counts the number of images and label files in the training and validation sets.


## Important notebooks

To refer to k-means creation of cluster (unsupervised classes, semi-supervised method) and output from semi-supervised methods, please refer to notebooks/coffeScale.ipynb, notebooks/yolov7+kmeans_detection_... ; 
Outputs from general yolo models can be found at notebooks/coffeAI-detector.ipynb;
