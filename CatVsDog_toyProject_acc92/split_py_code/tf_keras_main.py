import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import load_model
from statistics import mean
from json import dumps
import cv2
import random
import typing
# my own libraries
from training_plotter import NN_training_history_plotter
from nn_models import NN_models

#Specifying the image
Image_Width=360
Image_Height=360
Batch_Size = 32
Image_Size=(Image_Width,Image_Height)
Image_Channels=3
Model_Path: str = "dog_cat_adam_aug_CNN.h5"
History_Path: str = "CNN_training_history.json"

def fitted_model(model: keras.models.Sequential, lr: float, train_imgs: keras.preprocessing.image.DirectoryIterator, val_imgs: keras.preprocessing.image.DirectoryIterator) -> typing.Tuple:
    model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=["accuracy"])
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='weights.best.inc.blond.hdf5', verbose=1, save_best_only=True)
    model_history = model.fit(train_imgs, validation_data=val_imgs, steps_per_epoch=len(train_imgs), validation_steps=len(val_imgs), epochs=10, callbacks=[checkpointer])
    return (model, model_history)

# first 50% epochs: full lr, 50%-80% epochs: 25% lr, 80%-100% epochs: 5% lr (default)
def stepped_down_lr(initial_lr: float, total_epochs: int, round_index: int, epochs_per_round: int) -> float:
    assert total_epochs>=10 and total_epochs%10==0, "The number of datasets must be divsible by 10!"
    assert 0<(round_index*epochs_per_round)<=total_epochs, "The index of an epoch 'i' must suffice 0<i<=total_epochs!"
    return initial_lr if round_index*epochs_per_round<=total_epochs/2 else initial_lr/4 if round_index*epochs_per_round<=total_epochs*0.8 else initial_lr/20

if __name__ == '__main__':
    # image augmentation
    train_generator = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=45, fill_mode='nearest', 
        brightness_range = (0.8, 1.25),
        zoom_range= (0.8, 1.25),
        width_shift_range=0.2, height_shift_range=0.2,
        channel_shift_range=50
    )
    # non-augmented original image data flows
    original_generator = ImageDataGenerator(rescale=1./255)
    test_img_flow = original_generator.flow_from_directory(batch_size=Batch_Size, directory="../catdog_data/test", target_size=Image_Size, class_mode="binary")
    val_img_flow = original_generator.flow_from_directory(batch_size=Batch_Size, directory="../catdog_data/validation/", target_size=Image_Size, class_mode="binary")
    
    # set variables to store training history
    training_history: typing.Dict[str, typing.List[float]] = {}
    for his_key in ('loss', 'accuracy', 'val_loss', 'val_accuracy'):
        training_history[his_key] = [] 
    acc_per_100_epochs: typing.List[float] = []

    # start training
    # 100 epochs for each augmented dataset, maximally 500 epochs, stop training when the average validation accuracy of last 10 epochs becomes above 0.9
    models = NN_models(Image_size=Image_Size)
    model: keras.models.Sequential = models.He_uniform_4conv_CNN()
    stop_training: bool = False
    rounds_per_dataset: int = 10
    for dataset_count in range(1, 6):
        if stop_training: break
        train_img_flow = train_generator.flow_from_directory(batch_size=Batch_Size, directory="../catdog_data/train", target_size=Image_Size, class_mode="binary")
        for round_count in range(1, rounds_per_dataset+1):
            lr = stepped_down_lr(2e-3, 100, round_count, 10)
            print(f'Learning rate for this round is: {str(lr)}')
            model, history = fitted_model(model, lr, train_img_flow, val_img_flow)
            for his_key in history.history.keys():
                training_history[his_key].extend(history.history[his_key])
            sliding_avg_10_acc: float = mean(training_history['val_accuracy'][-10:])
            if sliding_avg_10_acc > 0.9: 
                stop_training = True   
                break
        acc = model.evaluate(test_img_flow, verbose=0)[1]
        print(f"Updated accuracy after using {str(dataset_count)} augmented training datasets: {round(acc, 4)*100}")
        model.save(Model_Path)
    # save training/validation history
    with open(History_Path, "w") as f:
        f.write(dumps(training_history))
    # model evaluation
    saved_model = load_model(Model_Path)
    acc = saved_model.evaluate(test_img_flow, verbose=0)[1]
    print(f"Final accuracy on the test dataset after {str(len(training_history['loss']))} epochs: {str(round(acc, 4)*100)}")
    # Visualization of training/validation loss/accuracy
    plotter = NN_training_history_plotter(History_Path)
    plotter.save_loss_fig('./CNN_300epochs_loss.png')
    plotter.save_acc_fig('./CNN_300epochs_acc.png')



    