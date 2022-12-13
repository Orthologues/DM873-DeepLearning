from json import loads
from matplotlib import pyplot as plt
import typing

History_Path: str = "../CNN_training_history.json"

# the class for visualization of our training
class NN_training_history_plotter():

    def __init__(self, jsonPath: str):
        self.jsonPath = jsonPath
        self.load_training_history()

    def load_training_history(self):
        with open(self.jsonPath, 'r') as f:
            saved_training_record = loads(f.readline().strip())
        t_loss: typing.List[float] = saved_training_record['loss']
        v_loss: typing.List[float] = saved_training_record['val_loss']
        t_acc: typing.List[float] = saved_training_record['accuracy']
        v_acc: typing.List[float] = saved_training_record['val_accuracy']
        assert len(t_loss)>=10 and len(t_loss)==len(v_loss)==len(t_acc)==len(v_acc), "There shall be at least ten training epochs and the lists that store training loss, validation loss, training accuracy, and validation accuracy must all have the same length!"
        self.t_loss, self.v_loss, self.t_acc, self.v_acc = t_loss, v_loss, t_acc, v_acc

    # Plot losses
    def save_loss_fig(self, saving_path: str):
        epochs = range(1, len(self.t_loss)+1)
        plt.figure(figsize=(16, 9), dpi=120)
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
        plt.figure(figsize=(16, 9), dpi=120)
        plt.rcParams['savefig.facecolor'] = 'white'
        plt.plot(epochs, self.t_acc, 'g.', label='training accuracy')
        plt.plot(epochs, self.v_acc, 'b.', label='validation accuracy')
        plt.title('Training and Validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(saving_path)


if __name__ == "__main__":
    plotter = NN_training_history_plotter(jsonPath=History_Path)
    plotter.save_loss_fig('../CNN_300epochs_loss.png')
    plotter.save_acc_fig('../CNN_300epochs_acc.png')