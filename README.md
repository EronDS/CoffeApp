# Implemented functions throughout the research paper Computer Vision-Aided Intelligent Monitoring of Coffee: Towards Sustainable Coffee Production (ERON, et al.), and back-end functions to CoffeApp mobile application


## Important custom functions implemented throghout the research

1. **classes_to_binary**: Takes the input folder from_folder, converts the class labels from multiclass (5 classes: green, green-yellow, cherry, raisin and dry) to binary (ripe and unripe) format using the mapping in the map dictionary, and saves the converted labels in the output folder to_folder;
2. **count_objects**: Counts the number of objects (coffe fruits) in the training and validation sets by iterating through the label files and counting the number of lines;
3. **classes_to_mono**: Similar to classes_to_binary, but converts the class labels to monochrome format (i.e., 0 for all classes, coffe fruits).
4. **split_data**: Splits the input folder into training and validation sets. The percentage of data to use for training is set by the split_percentage argument (default is 0.8);
5. **count_imgs_labels**: Counts the number of images and label files in the training and validation sets;
6. **unsupervised_annotation**: Create a semi-supervised (supervised locations and unsupervised classes, hence semi-supervised) annotation in yolo-format, where bounding boxes are human annoated (original dataset) and classes from k-means clusters (unsupervised classes);


## Important notebooks

To refer to k-means creation of cluster (unsupervised classes, semi-supervised method) and output from semi-supervised methods, please refer to notebooks/coffeScale.ipynb, notebooks/yolov7+kmeans_detection_... ; 
Outputs from general yolo models can be found at notebooks/coffeAI-detector.ipynb;

## Runs and Custom Yolov7 Functions

You can find the results of each of the selected models (YOLOv7, semi-supervised and supervised) in the runs/ folder. There you can see different metrics and batches for visualization. We do not share the weights because they have commercial value. We also provide custom functions based on the original YOLOv7 repository (https://github.com/WongKinYiu/yolov7).

## Data and Model Availabity

Our models and data have commercial value, so we do not fully make them publicly available (weights of the trained model, for example). If you are interested in using them, please contact the corresponding author of our paper with reasonable request.
