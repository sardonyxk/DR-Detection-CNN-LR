import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,)

np.random.seed(42)
tf.random.set_seed(42)

def build_cnn_feature_extractor(input_shape=(224, 224, 3))-> Sequential:
    model =Sequential(name="DR_CNN_Feature_Extractor")
    model.add(Input(shape=input_shape, name="input_layer"))
    
    #Convolutional Layer 1
    #32 filters
    #Expected output: (224, 224, 32)
    model.add(Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu",
        padding="same",
        name="conv_1"
    ))
    
    #Max Pooling Layer 1
    #Halves spatial dimensions
    #Expected output: (112, 112, 32)
    model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=2,
        name="maxpool_1"
    ))
    
    #Convolutional Layer 2
    #64 filters
    #Expected output: (112, 112, 64)
    
    model.add(Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu",
        padding="same",
        name="conv_2"
    ))
    #Max Pooling Layer 2
    #Expected output: (56, 56, 64)
    model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=2,
        name="maxpool_2"
    ))
    
    
    #Convolutional Layer 3
    #128 filters
    #Expected output: (56, 56, 128)
    model.add(Conv2D(
        filters=128,
        kernel_size=(3,3),
        activation="relu",
        padding="same",
        name="conv_3"
    ))
    
    #Max Pooling Layer 3
    #Expected output: (28, 28, 128)
    model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=2,
        name="maxpool_3"
    ))
    
    
    #Batch Normalization
    model.add(BatchNormalization(name="batchnorm"))
    
    #Flatten Layer
    #Expected output: (28*28*128,) = (100352,)
    model.add(Flatten(name="flatten"))
    return model

if __name__ == "__main__":
    
    #Build the CNN feature extractor model
    model = build_cnn_feature_extractor(input_shape=(224, 224, 3))
    
    #Print the model summary
    print("\n" + "=" * 65)
    print("CNN Feature Extractor Model Summary")
    print("=" * 65 + "\n")
    model.summary()
    
    
    expected_features = 28 * 28 * 128
    actual_features = model.output_shape[-1]
    
    print("\n" + "=" * 65)
    print(f" Expected feature vector size: {expected_features:,}")
    print(f" Actual feature vector size: {actual_features:,}")
    status = "Match" if actual_features == expected_features else "Mismatch"
    print(f" Status: {status}")
    print("=" * 65 + "\n")



