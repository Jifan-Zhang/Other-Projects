import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K

class Model_builder:
    """
    A model layers wrapper, building layer blocks.
    The instance variable remembers the layer structure.
        input_shape: The image shape CNN trains on
    """
    def __init__(self, input_shape = (512,512,3), output_shape = (512,512,3)):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.blocks = []

    def __bottle_neck_check__(f):
        def inner(self, *args, **kwargs):
            f(self, *args, **kwargs)
            x = self.blocks[-1]
            if(x.get_shape()[1]<self.output_shape[0] or x.get_shape()[2]<self.output_shape[1]):
                raise RuntimeError(f"The model has formed a bottle neck structure {(x.get_shape()[1],x.get_shape()[2])} < {(self.output_shape[0], self.output_shape[1])}, which should be recovered with up sampling, which is not implemented in the version.")
        return inner

    def input(self):
        out = tf.keras.Input(shape=self.input_shape)
        self.blocks.append(out)
        return out

    @__bottle_neck_check__
    def conv_block(self, n_filter=5, filter_size=(3,3), padding = "valid", strides=1):
        x = self.blocks[-1]
        out = tf.keras.layers.Conv2D(n_filter, filter_size, strides=strides, padding = padding, activation="relu")(x)
        self.blocks.append(out)
        return out

    @__bottle_neck_check__
    def pooling_block(self, strides=(2,2)):
        x = self.blocks[-1]
        out = tf.keras.layers.MaxPool2D()(x)
        self.blocks.append(out)
        return out

    @__bottle_neck_check__
    def conv_pool_block(self, n_filter=5, filter_size=(3,3), strides=1):
        x = self.blocks[-1]
        x = tf.keras.layers.Conv2D(n_filter, filter_size, strides=strides, padding = "same",activation="relu")(x)
        out = tf.keras.layers.MaxPool2D()(x)
        self.blocks.append(out)
        return out

    def top_block(self):
        x = self.blocks[-1]
        width = x.get_shape()[1]
        height = x.get_shape()[2]
        f_width = width + 1 - self.output_shape[0]
        f_height = height + 1 - self.output_shape[1]
        out = tf.keras.layers.Conv2D(3, (f_width,f_height) ,padding="valid", activation = "relu")(x)
        self.blocks.append(out)
        return out

    def build(self, monitor, lbd):
        """
        Create model
            monitor: list of unique positive integers. Indexing on which layer outputs to monitor the style.
            lbd: list of integers(with the same length as monitor) and the sum of this vector should be 1.
                lambda corresponds to the style loss.
        """
        if(len(monitor) != len(lbd)):
            raise RuntimeError("The lengths of monitor and lambda are not the same.")
        self.lbd = lbd
        outs=[]
        for i in range(len(self.blocks)):
            if i in monitor:
                outs.append(self.blocks[i])
                print(f"Monitoring -> {self.blocks[i].name}.")
        model = Model(inputs=self.blocks[0],outputs=outs)
        return model

    def get_loss(self, y_true, y_pred):
        """
        Implement Style Loss. Return one scalar,
        """
        loss = 0
        count = 0
        for pred,true in zip(y_pred,y_true):
            n_sample = y_pred.shape[0]
            width = y_pred.shape[1]
            height = y_pred.shape[2]
            n_channel = y_pred.shape[3]
            pred_gram = tf.transpose(y_pred,[0,3,2,1])
            pred_gram = tf.reshape(pred_gram, (n_sample,n_channel,width*height)) # Flatten last 2 dims for matmul
            pred_gram = tf.matmul(pred_gram, tf.transpose(pred_gram, [0,2,1]))
            
            true_gram = tf.transpose(y_pred,[0,3,2,1])
            true_gram = tf.reshape(true_gram, (n_sample,n_channel,width*height))
            true_gram = tf.matmul(true_gram, tf.transpose(true_gram, [0,2,1]))
            
            layer_loss = tf.reduce_sum(tf.matmul(pred_gram, true_gram))
            loss += layer_loss*self.lbd[count]
            count += 1
        return tf.matmul(loss, tf.transpose(self.lbd))