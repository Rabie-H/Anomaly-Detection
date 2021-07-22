Run train.py to retrieve data from .xlsx file, prepare training data and train both the classifier and the detector.

- To perform anomaly detection, we use a pre-trained Resnet50 to perform feature extraction. 
  We apply a PCA on the extracted features and use the output to train a one class classification SVM and an IsolationForest.
  The performances of the two models are compared, and the best algorithms is adopted.
  
- The multiclass classifier is a CNN network.

Run tkinter interface using the interface.py file.

- Select a sample Image
- Detect Anomaly
- If anomaly, detect the type of the anomaly.
