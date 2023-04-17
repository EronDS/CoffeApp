import os
import random
import shutil
from tqdm import tqdm


def classes_to_binary(from_folder:str, to_folder:str, map:dict) -> None:
    os.makedirs(to_folder, exist_ok = True)
    for file_dir in os.listdir(from_folder):
        with open(from_folder+'/'+file_dir, 'r') as f:
            new_file_dir = to_folder + '/' + file_dir
            with open(new_file_dir, 'w') as w:
                for line in f:
                    old_line = line 
                    new_line = map[line[0]] + old_line[1:]
                    w.write(new_line)
    return

def count_objects(folder:str) -> None:
    train, val = 0,0 
    for i in os.listdir(folder+'/train/labels/'):
        with open(folder+'/train/labels/' + i, 'r') as f:
            for line in f:
                train += 1
    for j in os.listdir(folder+'/val/labels'):
        with open(folder+'/val/labels/'+j,'r') as r:
            for line in r:
                val += 1
    print('Objects in Train Data: {}\nObjects in Val Data: {}\n'.format(train,val))

def classes_to_mono(from_folder:str, to_folder:str) -> None:
    os.makedirs(to_folder,exist_ok=True)
    for file_dir in os.listdir(from_folder):
        with open(from_folder + '/' + file_dir, 'r') as f:
            new_file_dir = to_folder + '/' + file_dir
            with open(new_file_dir, 'w') as w:
                for line in f:
                    old_line = line
                    new_line = '0'+old_line[1:]
                    w.write(new_line)
    return

def count_objects(folder:str) -> None:
    train, val = 0,0 
    for i in os.listdir(folder+'/train/labels/'):
        with open(folder+'/train/labels/' + i, 'r') as f:
            for line in f:
                train += 1
    for j in os.listdir(folder+'/val/labels'):
        with open(folder+'/val/labels/'+j,'r') as r:
            for line in r:
                val += 1
    print('Objects in Train Data: {}\nObjects in Val Data: {}\n'.format(train,val))

def split_data(folder_path:str, split_percentage:float = 0.8):
    """
    Splits a folder of images and corresponding labels into training and validation sets.

    Args:
        folder_path: Path to the folder containing images and labels.
        split_percentage: Percentage of data to use for training. The remaining percentage will be used for validation.

    Returns:
        train_folder: Path to the folder containing training images and labels.
        val_folder: Path to the folder containing validation images and labels.
    """
    # Create output directories
    train_folder = os.path.join(folder_path, 'train')
    val_folder = os.path.join(folder_path, 'val')
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_folder, 'labels'), exist_ok=True)
    train_folder_imgs,train_folder_lbls = train_folder + '/images/', train_folder + '/labels/'
    val_folder_imgs, val_folder_lbls = val_folder + '/labels/', train_folder + '/labels/'

    # Get list of images
    image_files = [f for f in os.listdir(os.path.join(folder_path, 'images'))]
    num_images = len(image_files)
    num_train = int(num_images * split_percentage)
    num_val = num_images - num_train

    # Shuffle images
    random.shuffle(image_files)

    # Copy images and labels to train folder
    for i in range(num_train):
        image_file = image_files[i]
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_image = os.path.join(folder_path, 'images', image_file)
        src_label = os.path.join(folder_path, 'labels', label_file)
        dst_image = os.path.join(train_folder, 'images', image_file)
        dst_label = os.path.join(train_folder, 'labels', label_file)
        shutil.copy(src_image, dst_image)
        shutil.copy(src_label, dst_label)

    # Copy images and labels to val folder
    for i in range(num_train, num_train + num_val):
        image_file = image_files[i]
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_image = os.path.join(folder_path, 'images', image_file)
        src_label = os.path.join(folder_path, 'labels', label_file)
        dst_image = os.path.join(val_folder, 'images', image_file)
        dst_label = os.path.join(val_folder, 'labels', label_file)
        shutil.copy(src_image, dst_image)
        shutil.copy(src_label, dst_label)

    return train_folder, val_folder


def count_imgs_labels(folder:str) -> None:
    train_imgs,val_imgs,train_lbls,val_lbls = 0,0,0,0    
    for i in os.listdir(folder+'/train/images'):
        train_imgs += 1
    for j in os.listdir(folder+'/val/images'): 
        val_imgs += 1 
    for u in os.listdir(folder + '/train/labels'):
        train_lbls += 1 
    for v in os.listdir(folder + '/val/labels'):
        val_lbls += 1
    print('Train Images: {} Train Labels: {}\n\
    Val Images: {} Val Labels: {}'.format(train_imgs,train_lbls, val_imgs,val_lbls))
    print('Total images: {} Total labels: {}'.format(train_imgs+val_imgs, train_lbls+val_lbls))

def yolo_format_to_yolov6(folder:str) -> None:
    imgs_train_folder,imgs_val_folder = folder+'/images/train', folder+'/images/val'
    os.makedirs(imgs_train_folder,exist_ok=True)
    os.makedirs(imgs_val_folder,exist_ok=True)

    lbls_train_folder,lbls_val_folder = folder+'/labels/train', folder+'/labels/val'
    os.makedirs(lbls_train_folder,exist_ok=True)
    os.makedirs(lbls_val_folder,exist_ok=True)

    # Training images
    for filename in tqdm(os.listdir(folder+'/train/images'), desc='Cópia de imagens de treinamento'):
        shutil.copy(os.path.join(folder, 'train/images', filename), imgs_train_folder)

    # Validation images
    for filename in tqdm(os.listdir(folder+'/val/images'), desc='Cópia de imagens de validação'):
        shutil.copy(os.path.join(folder, 'val/images', filename), imgs_val_folder)

    # Training labels
    for filename in tqdm(os.listdir(folder+'/train/labels'), desc='Cópia de arquivos de labels de treinamento'):
        if filename.endswith('.txt'):
            shutil.copy(os.path.join(folder, 'train/labels', filename), lbls_train_folder)

    # Validation labels
    for filename in tqdm(os.listdir(folder+'/val/labels'), desc='Cópia de arquivos de labels de validação'):
        if filename.endswith('.txt'):
            shutil.copy(os.path.join(folder, 'val/labels', filename), lbls_val_folder)
            

def unsupervised_annotation(_model,folder,k):
    '''
    create semi-supervised annotation in yolo_format,
    bouding_box location coming from supervised annotation, 
    classification coming from _model (kmeans)
    '''
    label_folder = folder + '/labels/'
    os.makedirs(folder + '/unsupervised_{}_label'.format(k), exist_ok=True)
    new_label_folder = folder + '/unsupervised_{}_label'.format(k)
    img_folder = folder + '/images/'

    for img_file in os.listdir(img_folder):
        name, extension = os.path.splitext(img_file)
        text_file = os.path.join(label_folder, name + '.txt')
        new_text_file = os.path.join(new_label_folder, name + '.txt')
        img_path = os.path.join(img_folder, img_file) 

        # Load the image
        img = cv2.imread(img_path)

        # Read the label file
        with open(text_file, 'r') as f:
            lines = f.readlines()

        # Extract the bounding box coordinates and labels for each object
        objects = []
        for line in lines:
            # Convert the label line to a list of values
            label_list = line.strip().split(' ')

            # Extract the bounding box coordinates
            x_center = float(label_list[1])
            y_center = float(label_list[2])
            width = float(label_list[3])
            height = float(label_list[4])

            # Calculate the top-left and bottom-right coordinates of the bounding box
            x_top_left = int((x_center - (width / 2)) * img.shape[1])
            y_top_left = int((y_center - (height / 2)) * img.shape[0])
            x_bottom_right = int((x_center + (width / 2)) * img.shape[1])
            y_bottom_right = int((y_center + (height / 2)) * img.shape[0])

            # Crop the region corresponding to the bounding box and resize to 28x28x3
            object_img = img[y_top_left:y_bottom_right, x_top_left:x_bottom_right]
            
            if object_img.size == 0:
                continue  # skip empty image
            try:
                object_img = cv2.resize(object_img, (28,28))
            except cv2.error as e:
                print(f"Error: {e}")
                continue  # skip invalid image
            

            cv2_imshow(object_img)
            object_img_rgb = cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB)
            object_img_lab = cv2.cvtColor(object_img_rgb, cv2.COLOR_RGB2LAB)
            img_ab = object_img_lab[:, :, 1:]  # extract a,b channels from LAB image
            img_ab_vect = np.reshape(img_ab, (28*28*2))  # vectorize a,b channels
            img_ab_vect = img_ab_vect.astype('float32')  # convert to float32
            img_ab_vect /= 255.0  # normalize to [0, 1]
            img_ab_vect = np.array(img_ab_vect)

            # Add the object and its label to the list
            label = int(label_list[0])
            objects.append((img_ab_vect.reshape(1,-1), label, x_top_left, y_top_left, x_bottom_right, y_bottom_right))

        # Make predictions for the objects using the given model
        predictions = [] 
        for obj in objects:
            vec = obj[0]
            vec = np.array(vec)
            prediction = _model.predict(vec)
            predictions.append(prediction)


        # Write the predicted labels to the label file
        with open(new_text_file, 'w') as f:
            for i, obj in enumerate(objects):
                label = predictions[i]
                x_center = ((obj[2] + obj[4]) / 2) / img.shape[1]
                y_center = ((obj[3] + obj[5]) / 2) / img.shape[0]
                width = (obj[4] - obj[2]) / img.shape[1]
                height = (obj[5] - obj[3]) / img.shape[0]
                f.write(f"{label} {x_center} {y_center} {width} {height}\n")
