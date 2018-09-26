from keras import backend as K
from keras.callbacks import TensorBoard

class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  # add other arguments to __init__ if you need
        super(LRTensorBoard, self).__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super(LRTensorBoard, self).on_epoch_end(epoch, logs)
