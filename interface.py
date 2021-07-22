import pickle
import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
from keras.models import load_model
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications.resnet50 import ResNet50
from utils import read_and_prep_images, preprocess_image
from functools import partial
import numpy as np




class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.hi_there = tk.Button(self)
        self.hi_there["text"] = "Browse Files"
        self.hi_there["command"] = self.browse_files
        self.hi_there.pack(side="top")

        self.quit = tk.Button(self, text="QUIT", fg="red",
                              command=self.master.destroy)
        self.quit.pack(side="bottom")

    def browse_files(self):
        filename = filedialog.askopenfilename(initialdir="/",
                                              title="Select a File",
                                              filetypes=(("Image files",
                                                          "*.jpg*"),
                                                         ("all files",
                                                          "*.*")))

        print(filename)
        img = Image.open(filename)
        img = img.resize((250, 250), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(root, image=img)
        panel.image = img
        panel.pack(side="bottom")

        self.detect = tk.Button(self)
        self.detect["text"] = "Detect"
        self.detect["command"] = partial(self.detect_and_classify, filename)
        self.detect.pack(side="bottom")

    def detect_and_classify(self, filename):
        with open('best_clf.pkl', 'rb') as handler:
            detector = pickle.load(handler)
        classifier = load_model('CNN_aug_best_weights.h5')
        img_height = 349
        img_width = 259
        X_test = read_and_prep_images([filename])
        resnet_model = ResNet50(input_shape=(img_height, img_width, 3), weights='imagenet', include_top=False,
                                pooling='avg')  # Since top layer is the fc layer used for predictions

        X_test = resnet_model.predict(X_test)

        # Apply standard scaler to output from resnet50
        with open('scaler.pkl', 'rb') as handler:
            ss = pickle.load(handler)
        X_test = ss.transform(X_test)

        # Take PCA to reduce feature space dimensionality
        with open('pca.pkl', 'rb') as handler:
            pca = pickle.load(handler)
        X_test = pca.transform(X_test)
        detection = detector.predict(X_test)
        print(detection)
        if detection[0] == 1:
            detection_text = 'Anomaly Detected'
            self.detection_text = tk.Label(self, text=detection_text)
            self.detection_text.pack(side="bottom")
            test_preprocessed_images = np.vstack([preprocess_image(fn) for fn in [filename]])
            array = classifier.predict(test_preprocessed_images, batch_size=1, verbose=1)
            classification = np.argmax(array, axis=1)
            print(classification)
            self.classification_text = tk.Label(self, text="Anmomaly Class {}".format(classification[0]))
            self.classification_text.pack(side="bottom")
        else:
            detection_text = 'Anomaly Not Detected'
            self.detection_text = tk.Label(self, text=detection_text)
            self.detection_text.pack(side="bottom")





if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("500x500")
    app = Application(master=root)
    app.mainloop()
