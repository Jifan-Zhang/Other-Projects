import tensorflow as tf
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import numpy as np
from Generator import Generate

class Model_builder:
    """
    A model layers wrapper, building layer blocks.
    The instance variable remembers the layer structure.
        input_shape: The image shape CNN trains on
    """
    def __init__(self, input_shape = (1024,1024,3), output_shape = (512,512,3)):
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
            monitor: list of unique positive integers. Indexing on which intermediate layer outputs to monitor the style (except the last layer).
            lbd: list of integers(with the same length as monitor) and the sum of this vector should be 1.
                lambda corresponds to the style loss.
        """
        if(len(monitor) != len(lbd)):
            raise RuntimeError("The lengths of monitor and lambda are not the same.")
        
        # The last layer loss accounts for highest (0.5) loss weight.
        lbd = [0.5*x for x in lbd] + [0.5]
        self.lbd = lbd
        outs=[]
        for i in range(len(self.blocks)):
            if i in monitor:
                outs.append(self.blocks[i])
                print(f"Monitoring -> {self.blocks[i].name}.")

        outs.append(self.blocks[-1])
        self.model = Model(inputs=self.blocks[0],outputs=outs)
        return self.model

    def __loss__(self, y_true, y_pred):
        """
        Implement Style Loss. Method returns a tf scalar.
        """
        loss = tf.Variable(0, dtype=tf.float32)
        count = 0
        for pred,true in zip(y_pred,y_true):
            n_sample = pred.shape[0]
            width = pred.shape[1]
            height = pred.shape[2]
            n_channel = pred.shape[3]
            pred_gram = tf.transpose(pred,[0,3,2,1])
            pred_gram = tf.reshape(pred_gram, (n_sample,n_channel,width*height)) # Flatten last 2 dims for matmul
            pred_gram = tf.matmul(pred_gram, tf.transpose(pred_gram, [0,2,1]))
            
            true_gram = tf.transpose(true,[0,3,2,1])
            true_gram = tf.reshape(true_gram, (n_sample,n_channel,width*height))
            true_gram = tf.matmul(true_gram, tf.transpose(true_gram, [0,2,1]))
            
            layer_loss = tf.reduce_sum(tf.math.square(pred_gram-true_gram))/(2*width*height)**2
            loss = loss + layer_loss*self.lbd[count]
            count += 1
        return loss

    def __get_pred__(self, path="color-gradient.jpg"):
        new_img = next(Generate(path, self.input_shape))
        y_true = self.model(tf.expand_dims(new_img,axis=0))
        img_input = tf.random.uniform(minval=-1, maxval=1, shape=(1,)+(self.input_shape),dtype=tf.float32)
        y_pred = self.model(img_input)
        return (y_true,y_pred)

    def get_gradient(self):
        with tf.GradientTape() as tape:
            loss = self.__loss__(*self.__get_pred__())
            self.__current_loss__ = loss
        return tape.gradient(loss, self.model.trainable_weights)