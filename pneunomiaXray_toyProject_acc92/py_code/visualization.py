"""
Written by Jiawei Zhao on 12th of December, 2022 to implemented customized Keras layers 
"""

from matplotlib import pyplot as plt
from typing import List, Tuple
import json
from keras.models import load_model
from numpy import ndarray, array
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K 
from pandas import read_csv, DataFrame
import cv2
import os
import seaborn as sns
from customized_layers import *
from training_main import History_Path, Model_Path, process_input_data


class NN_model_plotter():

    def __init__(self, json_path: str = History_Path, model_path:str = Model_Path):
        self.json_path = json_path
        self.model_path = model_path
        self.load_training_history()

    # load the training history (a .json file)
    def load_training_history(self):
        with open(self.json_path, 'r') as f:
            saved_training_record = json.loads(f.readline().strip())
        t_loss: List[float] = saved_training_record['loss']
        v_loss: List[float] = saved_training_record['val_loss']
        t_acc: List[float] = saved_training_record['accuracy']
        v_acc: List[float] = saved_training_record['val_accuracy']
        assert len(t_loss)>=10 and len(t_loss)==len(v_loss)==len(t_acc)==len(v_acc), "There shall be at least ten training epochs and the lists that store training loss, validation loss, training accuracy, and validation accuracy must all have the same length!"
        self.t_loss, self.v_loss, self.t_acc, self.v_acc = t_loss, v_loss, t_acc, v_acc

    # Plot losses
    def save_loss_fig(self, saving_path: str):
        epochs = range(1, len(self.t_loss)+1)
        plt.figure(figsize=(8, 6), dpi=240)
        plt.rcParams['savefig.facecolor']='white'
        plt.plot(epochs, self.t_loss, 'g.', label='training loss')
        plt.plot(epochs, self.v_loss, 'b.', label='validation loss')
        plt.title('Training and Validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(saving_path)

    # Plot accuracies
    def save_acc_fig(self, saving_path: str):
        epochs = range(1, len(self.t_loss)+1)
        plt.figure(figsize=(8, 6), dpi=240)
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.plot(epochs, self.t_acc, 'g.', label='training accuracy')
        plt.plot(epochs, self.v_acc, 'b.', label='validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(saving_path)

    # Plot confusion matrix
    def plot_confusion_matrix(self, saving_path: str):
        model = load_model(self.model_path, custom_objects={
            'MyConv2D': MyConv2D,
            'MyMaxPool2D': MyMaxPool2D,
            'MyDense': MyDense
        })
        test_imgs, test_labels = read_test_data()
        test_imgs/=255
        test_pred: ndarray[bool] = model.predict(test_imgs, batch_size=32)
        test_pred: ndarray[bool] = (test_pred > 0.5)
        conf_mat = confusion_matrix(test_labels, test_pred)
        # visualizing the confusion matrix
        df1 = DataFrame(columns=["Predicted Healthy","Predicted Pneumonia"], index= ["Actual Healthy","Actual Pneumonia"], data=conf_mat) 
        f, ax = plt.subplots(figsize=(16, 9), dpi=120)
        plt.rcParams.update({'font.size': 18})
        sns.heatmap(df1, annot=True, cmap="Blues", fmt= '.0f', ax=ax)
        plt.xlabel("Predicted Label")
        plt.xticks(size = 12, rotation=-10)
        plt.yticks(size = 12, rotation = 15)
        plt.ylabel("True Label")
        plt.title("Confusion Matrix", size = 15)
        plt.savefig(saving_path)
        print("True Negative:" , (conf_mat[0,0]))
        print("True Positive:" , (conf_mat[1,1]))
        print("False Positive:" , (conf_mat[0,1]))
        print("False Negative:" , (conf_mat[1,0]))


def read_test_data(input_path: str = "../") -> Tuple[array]:  
    test_data: List[ndarray] = []
    test_labels: List[int] = []

    for cond in ('/NORMAL/', '/PNEUMONIA/'):
        for img_fname in (os.listdir(f"{input_path}test_encoded{cond}")):
            """
            Reads the text files into numpy arrays.
            PNG images are returned as float arrays (0-1). 
            All other formats are returned as int arrays, with a bit depth determined by the file's contents.
            """
            txt_fname = f"{input_path}test_encoded{cond}{img_fname}"
            raw_img_array = read_csv(txt_fname, header=None, sep=" ").to_numpy()
            greyscale_img_array = cv2.resize(raw_img_array, (224, 224))
            # converts one-channel images into three channel images
            rgb_img_array = cv2.merge([greyscale_img_array]*3)
            rgb_img_array = rgb_img_array.astype('float32')
            label = 0 if cond=="/NORMAL/" else 1
            test_data.append(rgb_img_array)
            test_labels.append(label)

    test_data = array(test_data)
    test_labels = array(test_labels)
    return (test_data, test_labels)


if __name__ == "__main__":
    # Set the learning phase to Test since the model is already trained.
    K.set_learning_phase(0)
    plotter = NN_model_plotter()
    plotter.plot_confusion_matrix("../visualization/conf_mat.png")
    plotter.save_acc_fig("../visualization/acc_90epochs.png")
    plotter.save_loss_fig("../visualization/loss_90epochs.png")
