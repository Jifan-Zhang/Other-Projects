import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Model import Model_builder
from Generator import Generate
import numpy as np

if __name__ == "__main__":
    # Future edit: get command line input
    input_shape = (1024,1024, 3)
    output_shape = (256, 256, 3)
    builder = Model_builder(input_shape, output_shape)
    builder.input()
    builder.conv_block(16, filter_size=(5,5), padding="same")
    builder.conv_block(16, filter_size=(7,7), padding="same")
    builder.pooling_block((2,2))
    builder.conv_block(32, filter_size=(9,9), padding="same")
    builder.conv_block(32, filter_size=(11,11), padding="same")
    builder.pooling_block((2,2))
    builder.top_block()
    model = builder.build([2,5],[0.5,0.5])


    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    gen = Generate(path="color-gradient.jpg", out_shape=input_shape)
    for i in range(40):
        grads = builder.get_gradient()
        optimizer.apply_gradients(zip(grads, builder.model.trainable_weights))
        print(f"Loss at step {i} = {builder.__current_loss__}")


    pic = np.expand_dims(next(gen),axis=0)
    builder.model.predict(pic)
