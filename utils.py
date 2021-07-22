import random
import uuid
import os
from shutil import copyfile
import numpy as np
import openpyxl
from matplotlib import pyplot as plt
from openpyxl_image_loader import SheetImageLoader
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import pickle


img_height = 349
img_width = 259
image_rows = [i for i in range(4, 26, 3)]
image_cloumns = ['D', 'F', 'H', 'J', 'L', 'N']
anomaly_column = 'B'
image_number = [3, 3, 6, 6, 2, 3, 1, 1]


def prepare_excel_content(excel_path, sheet_name, data_folder_name):
    # loading the Excel File and the sheet
    pxl_doc = openpyxl.load_workbook(excel_path)
    sheet = pxl_doc[sheet_name]
    image_loader = SheetImageLoader(sheet)
    for index, row in enumerate(image_rows):
        anomaly = sheet[anomaly_column + str(row)].value
        path = os.path.join(os.getcwd(), data_folder_name, anomaly)
        if not os.path.isdir(path):
            os.mkdir(path)
        for im_idx, image_column in enumerate(image_cloumns[:image_number[index]]):
            try:
                image = image_loader.get(image_column + str(row))
                image.save(os.path.join(path, 'image' + str(im_idx) + '.jpg'))
            except:
                print(anomaly + ' : cell ' + image_column + str(row) + ' caused an error.')


def prepare_one_class_classification_train_data(class_folders_path, destination_folder_path):
    # Create new folder to store all data
    if not os.path.isdir(destination_folder_path):
        os.mkdir(destination_folder_path)
    # iterate over all folders and paste content into the new folder
    for class_name in os.listdir(class_folders_path):
        class_folder = os.path.join(class_folders_path, class_name)
        for image_name in os.listdir(class_folder):
            orig_image_path = os.path.join(class_folder, image_name)
            dest_image_path = os.path.join(destination_folder_path, image_name)
            if os.path.isfile(dest_image_path):
                # Generate a new filename if the same filename already exists
                image_name = str(uuid.uuid4()) + os.path.splitext(image_name)[-1]
                print(image_name)
                dest_image_path = os.path.join(destination_folder_path, image_name)
            copyfile(orig_image_path, dest_image_path)


def read_and_prep_images(img_paths, img_height=img_height, img_width=img_width):
    imgs = [load_img(img_path, target_size=(img_height, img_width)) for img_path in img_paths]
    img_array = np.array([img_to_array(img) for img in imgs])
    # output = img_array
    output = preprocess_input(img_array)
    return output


def train_anomaly_detector(train_img_paths):
    # split cars data into train, test, and val
    train_img_paths, test_img_paths = train_test_split(train_img_paths, test_size=0.25, random_state=42)

    X_train = read_and_prep_images(train_img_paths)
    X_test = read_and_prep_images(test_img_paths)
    # get features from resnet50

    # X : images numpy array
    resnet_model = ResNet50(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False,
                            pooling='avg')  # Since top layer is the fc layer used for predictions

    X_train = resnet_model.predict(X_train)
    X_test = resnet_model.predict(X_test)

    # Apply standard scaler to output from resnet50
    ss = StandardScaler()
    ss.fit(X_train)
    with open('scaler.pkl', 'wb') as handler:
        pickle.dump(ss, handler)
    X_train = ss.transform(X_train)

    # Take PCA to reduce feature space dimensionality
    pca = PCA(n_components=256, whiten=True)
    pca = pca.fit(X_train)
    with open('pca.pkl', 'wb') as handler:
        pickle.dump(pca, handler)
    print('Explained variance percentage = %0.2f' % sum(pca.explained_variance_ratio_))
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)

    # Train classifier and obtain predictions for OC-SVM
    oc_svm_clf = svm.OneClassSVM(gamma=0.001, kernel='rbf', nu=0.08)
    if_clf = IsolationForest(contamination=0.08, max_features=1.0, max_samples=1.0, n_estimators=40)

    oc_svm_clf.fit(X_train)
    if_clf.fit(X_train)

    oc_svm_preds = oc_svm_clf.predict(X_test)
    if_preds = if_clf.predict(X_test)

    oc_svm_acc = oc_svm_preds.sum() / len(X_test)
    iso_forest_acc = if_preds.sum() / len(X_test)
    print('oc_svm acc : %0.2f' % oc_svm_acc)
    print('iso_forest acc : %0.2f' % iso_forest_acc)
    if oc_svm_acc > iso_forest_acc:
        with open('best_clf.pkl', 'wb') as handler:
            pickle.dump(oc_svm_clf, handler)
        return oc_svm_clf
    else:
        with open('best_clf.pkl', 'wb') as handler:
            pickle.dump(if_clf, handler)
        return if_clf


def split_data(SOURCE, TRAINING, VALIDATION, SPLIT_SIZE):
    files = []
    for filename in os.listdir(SOURCE):
        file = os.path.join(SOURCE, filename)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    training_length = int(len(files) * SPLIT_SIZE)
    valid_length = int(len(files) - training_length)
    shuffled_set = random.sample(files, len(files))
    training_set = shuffled_set[0:training_length]
    valid_set = shuffled_set[training_length:]

    for filename in training_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(TRAINING, filename)
        copyfile(this_file, destination)

    for filename in valid_set:
        this_file = os.path.join(SOURCE, filename)
        destination = os.path.join(VALIDATION, filename)
        copyfile(this_file, destination)


def train_anomaly_classifier(original_data_path, destination_data_path):
    class_names = os.listdir(original_data_path)
    batch_size = 2
    train_dir = os.path.join(destination_data_path, 'train')
    validation_dir = os.path.join(destination_data_path, 'validation')
    # if folder doesn't exist already
    if not os.path.isdir(destination_data_path):
        os.mkdir(destination_data_path)
        # Create two sub-folders
        os.mkdir(train_dir)
        os.mkdir(validation_dir)
        for class_name in class_names:
            train_class_folder = os.path.join(train_dir, class_name)
            os.mkdir(train_class_folder)
            val_class_folder = os.path.join(validation_dir, class_name)
            os.mkdir(val_class_folder)
            split_data(SOURCE=os.path.join(original_data_path, class_name),
                       TRAINING=train_class_folder,
                       VALIDATION=val_class_folder,
                       SPLIT_SIZE=.85)
            print('Training {} images are: '.format(class_name) + str(len(os.listdir(train_class_folder))))
            print('Valid {} images are: '.format(class_name) + str(len(os.listdir(validation_dir))))

    train_datagen = ImageDataGenerator(rescale=1 / 255.0,
                                       rotation_range=30,
                                       zoom_range=0.4,
                                       horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        target_size=(img_height, img_width))
    validation_datagen = ImageDataGenerator(rescale=1 / 255.0)

    validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                                  batch_size=batch_size,
                                                                  class_mode='categorical',
                                                                  target_size=(img_height, img_width)
                                                                  )
    model = Sequential([
        Conv2D(16, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)), MaxPooling2D(2, 2),
        Conv2D(32, (3, 3), activation='relu'), MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(256, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(8, activation='softmax')
    ])
    model.summary()
    callbacks = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    # autosave best Model
    best_model_file = 'CNN_aug_best_weights.h5'
    best_model = ModelCheckpoint(best_model_file, monitor='val_accuracy', verbose=1, save_best_only=True)
    model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics = ["accuracy"])
    history = model.fit(train_generator,
                                  epochs=30,
                                  verbose=1,
                                  validation_data=validation_generator,
                                  callbacks=[best_model]
                                  )
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    fig = plt.figure(figsize=(14, 7))
    plt.plot(epochs, acc, 'r', label="Training Accuracy")
    plt.plot(epochs, val_acc, 'b', label="Validation Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.show()

    fig2 = plt.figure(figsize=(14, 7))
    plt.plot(epochs, loss, 'r', label="Training Loss")
    plt.plot(epochs, val_loss, 'b', label="Validation Loss")
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and validation loss')

    return model


def preprocess_image(path):
    img = load_img(path, target_size = (img_height, img_width))
    a = img_to_array(img)
    a = np.expand_dims(a, axis = 0)
    a /= 255.
    return a