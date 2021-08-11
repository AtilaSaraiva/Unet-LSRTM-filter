from keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from keras.optimizers import Adam
from keras.models import *
from keras.utils import plot_model

def unet(pretrained_weights = None, input_size = (512,512,1), learningRate=0.5e-4):
    inputs = Input(input_size)
    # Conv1a 32 3x3/1
    # Conv1b 32 3x3/1
    # Maxpool1 2x2/2
    conv1a = Conv2D(32, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1b = Conv2D(32, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv1a)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1b)

    # Conv2a 64 3x3/1
    # Conv2b 64 3x3/1
    # Maxpool2 2x2

    conv2a = Conv2D(64, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2b = Conv2D(64, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv2a)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2b)

    # Conv3a 128 3x3/1
    # Conv3b 128 3x3/1
    # Maxpool3 2x2

    conv3a = Conv2D(128, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv3b = Conv2D(128, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv2a)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3b)

    # Conv4a 128 3x3/1
    # Conv4b 128 3x3/1
    # Up-Conv1 2x2
    # Concatenate (Up-Conv1,conv3b)

    conv4a = Conv2D(128, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4b = Conv2D(128, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv4a)
    up1 = Conv2D(128, 2, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv4b))
    merge1 = concatenate([up1,conv3b], axis = 3)

    # Conv5a 128 3x3/1
    # Conv5b 128 3x3/1
    # Up-Conv2 2x2
    # Concatenate (Up-Conv2,conv2b)

    conv5a = Conv2D(128, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(merge1)
    conv5b = Conv2D(128, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv5a)
    up2 = Conv2D(128, 2, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv5b))
    merge2 = concatenate([up1,conv2b], axis = 3)

    # Conv6a 64 3x3/1
    # Conv6b 64 3x3/1
    # Up-Conv3
    # Concatenate (Up-Conv3,conv1b)

    conv6a = Conv2D(64, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(merge2)
    conv6b = Conv2D(64, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv5a)
    up3 = Conv2D(64, 2, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6b))
    merge3 = concatenate([up3,conv1b], axis = 3)

    # Conv7a 32 3x3/1
    # Conv7b 32 3x3/1
    # Conv7 1 3x3/1

    conv7a = Conv2D(32, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(merge3)
    conv7b = Conv2D(32, 3, activation = 'tanh', padding = 'same', kernel_initializer = 'he_normal')(conv7a)
    conv7c = Conv2D(1, 3, activation = 'linear', padding = 'same', kernel_initializer = 'he_normal')(conv7b)

    model = Model(inputs = inputs, outputs = conv7c)

    model.compile(optimizer = Adam(learning_rate = learningRate),
            loss = 'mean_squared_error',
            metrics = ['mean_squared_error'])

    if(pretrained_weights):
        model.load_weights(pretrained_weights)

    return model
