"""
Written by Jiawei Zhao on 12th of December, 2022 to implemented customized Keras layers 
"""

#Specifying the image
Image_Width=224
Image_Height=224
Batch_Size = 32
Image_Size=(Image_Width,Image_Height)
Image_Channels=3
History_Path = "../self_CNN_training_history.json"
Model_Path = "../pneumonia_aug_self_CNN.h5"

import json
import os
from typing import *
from customized_layers import *
from pandas import read_csv
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from statistics import mean
from keras.models import Sequential, save_model, load_model
import tensorflow as tf
import keras
import keras.backend as K 
from keras.optimizers import Adam
from model import create_self_CNN_Model


"""
Method for loading train images and test images from disk storage into a 
"tf.keras.preprocessing.image.DirectoryIterator" rtype
and
"tf.keras.preprocessing.image.NumpyArrayIterator" rtype
respectively
"""
def process_input_data(
    input_path:str="", 
    img_width:int=Image_Width, img_height:int=Image_Height, img_channels:int=Image_Channels,
    batch_size:int=Batch_Size
) -> Tuple[any]:
    
    # input parameter assertions
    assert img_width >= 16 and img_height>=16
    assert img_channels in {1, 3, 4}
    assert 16<=batch_size<=256
    
    # ImageDataGenerator objects
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=45, fill_mode='nearest', 
        brightness_range = (0.8, 1.25),
        zoom_range= (0.8, 1.25),
        width_shift_range=0.2, height_shift_range=0.2,
        channel_shift_range=50
    )
    
    # only rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # This is fed to the network in the specified batch sizes and image dimensions
    train_iterator = train_datagen.flow_from_directory(
        directory=input_path+'train', 
        target_size=(img_width, img_height), 
        batch_size=batch_size, 
        class_mode='binary', 
        shuffle=True
    )

    # I will be making predictions off of the test set in one batch size
    # This is useful to be able to get the confusion matrix
    test_data: List[np.ndarray] = []
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
            greyscale_img_array = cv2.resize(raw_img_array, (img_width, img_height))
            # converts one-channel images into three channel images
            rgb_img_array = cv2.merge([greyscale_img_array]*3)
            rgb_img_array = rgb_img_array.astype('float32')
            label = 0 if cond=="/NORMAL/" else 1
            test_data.append(rgb_img_array)
            test_labels.append(label)
        
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)
    
    test_iterator = test_datagen.flow(
        test_data, test_labels, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    return train_iterator, test_iterator


"""
Method for fitting and updating the model
"""
def fitted_own_model(model: Sequential, lr: float, train_imgs: keras.preprocessing.image.DirectoryIterator, val_imgs: keras.preprocessing.image.NumpyArrayIterator) -> Tuple:
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=["accuracy"])
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights.best.inc.blond.hdf5', verbose=1, save_best_only=True)
    model_history = model.fit(train_imgs, validation_data=val_imgs, steps_per_epoch=len(train_imgs), validation_steps=len(val_imgs), epochs=10, callbacks=[checkpointer])
    return (model, model_history)


"""
Method for updating the learning rate
"""
def obtain_lr(initial_lr: float, total_epochs: int, round_index: int, epochs_per_round: int) -> float:
    assert total_epochs>=10 and total_epochs%10==0, "The number of datasets must be divsible by 10!"
    assert 0<(round_index*epochs_per_round)<=total_epochs
    return initial_lr if round_index*epochs_per_round<=total_epochs/2 else initial_lr/4 if round_index*epochs_per_round<=total_epochs*0.8 else initial_lr/20


"""
trains "create_self_CNN_Model()" here
utilizes all the aforementioned methods
"""
if __name__ == "__main__":
    # train/test image loading
    train_iterator, test_iterator = process_input_data(input_path="../")
    
    # define the dict to store training history
    training_history: Dict[str, List[float]] = {}
    for his_key in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
        training_history[his_key] = [] 
    acc_per_100_epochs: List[float] = []

    # define the customized CNN model
    own_model = create_self_CNN_Model(image_size=Image_Size)
    own_model.build((Batch_Size, Image_Width, Image_Height, Image_Channels))
    own_model.summary()

    # proper training
    # 100 epochs for each augmented dataset, maximally 500 epochs, stop training when the average validation accuracy of last 10 epochs becomes above 0.9
    stop_training: bool = False
    rounds_per_dataset: int = 10
    for dataset_count in range(1, 6):
        if stop_training: break
        for round_count in range(1, rounds_per_dataset+1):
            lr = obtain_lr(2e-3, 100, round_count, 10)
            print(f'Learning rate for this round is: {str(lr)}')
            own_model, history = fitted_own_model(own_model, lr, train_iterator, test_iterator)
            for his_key in history.history.keys():
                training_history[his_key].extend(history.history[his_key])
            save_model(own_model, Model_Path)
            sliding_avg_10_acc: float = mean(training_history['val_accuracy'][-10:])
            if sliding_avg_10_acc > 0.9: 
                stop_training = True   
                break

    acc = own_model.evaluate(test_iterator, verbose=0)[1]
    print(f"Updated accuracy after using {str(dataset_count)} augmented training datasets: {round(acc, 4)*100}")
    save_model(own_model, Model_Path)
    
    # save training history into a .json file at disk storage
    with open(History_Path, "w") as f:
        f.write(json.dumps(training_history))

    # reload the training history from a .json file
    training_history, own_model = None, None
    K.set_learning_phase(0)
    own_model = load_model(Model_Path, custom_objects={
        'MyConv2D': MyConv2D,
        'MyMaxPool2D': MyMaxPool2D,
        'MyDense': MyDense
    })
    acc = own_model.evaluate(test_iterator, verbose=0)[1]*100
    with open(History_Path, "r") as f:
        training_history = json.loads(f.read())
        print(f"Number of trained epochs until we obtained an accuracy at {round(acc, 2)}: {len(training_history['val_accuracy'])}")
