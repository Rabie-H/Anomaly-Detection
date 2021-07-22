import os
import utils
from data_augment import perform_data_augmentation

data_folder_name = 'Data'
data_path = os.path.join(os.getcwd(), data_folder_name)

anomaly_detection_train_folder_name = 'anomaly_detection_train_data'
anomaly_detection_train_folder_path = os.path.join(os.getcwd(), anomaly_detection_train_folder_name)

anomaly_clf_train_folder_name = 'anomaly_clf_train_data'
anomaly_clf_train_folder_name_path = os.path.join(os.getcwd(), anomaly_clf_train_folder_name)

train_img_paths = [os.path.join(anomaly_detection_train_folder_path, filename) for filename in os.listdir(anomaly_detection_train_folder_path)]

if __name__ == '__main__':
    print('!===================Retrieving Data from Excel Sheet===================!')
    utils.prepare_excel_content(excel_path='Task.xlsx', sheet_name='Sample', data_folder_name=data_folder_name)
    print('!===================Performing Data Augmentation===================!')
    perform_data_augmentation(data_path, plot=False)
    print('!===================Preparing Anomaly Detector Training Data===================!')
    utils.prepare_one_class_classification_train_data(class_folders_path=data_folder_name, destination_folder_path=anomaly_detection_train_folder_path)
    print('!===================Training Anomaly Detector===================!')
    best_detector = utils.train_anomaly_detector(train_img_paths)
    print('!===================Training Anomaly Classifier===================!')
    best_clf = utils.train_anomaly_classifier(original_data_path= data_path, destination_data_path=anomaly_clf_train_folder_name_path)
    print('!===================Models are Successfully Trained===================!')