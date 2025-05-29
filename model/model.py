import keras
from keras import layers
from keras.models import model_from_json, Model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv3D,Conv1D, MaxPooling3D, BatchNormalization, Dropout, GlobalAveragePooling3D, Concatenate
from keras.layers import Input, Lambda, Dense, Flatten
from keras.applications import MobileNetV2, VGG16, InceptionV3
from keras.layers import (
    Input, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D, Concatenate, Add
)
def FineTuned_CNN_Model(IMG_SIZE: int, path:str, model=None) -> Model:
    """
    Create a batch-compatible CNN model using a pre-trained VGG model with custom weights.

    Args:
        IMG_SIZE (int): The height and width of the input images.

    Returns:
        keras.Model: A feature extractor model using custom weights.
    """
    model = keras.models.load_model(model, compile=False)
    print("Loaded custom weights into the VGG model")


    # Extract the feature map from the appropriate intermediate layer
    intermediate_layer = model.layers[-6].output

    # Ensure the intermediate layer output is connected to the new inputs
    feature_extractor = Model(inputs=model.input, outputs=intermediate_layer)

    # Freeze all layers in the feature extractor
    for layer in feature_extractor.layers:
        layer.trainable = False

    return feature_extractor


def CNN_Model(IMG_SIZE: int, preprocess_input) -> keras.Model:
    """
    Create a CNN model using InceptionV3 as a feature extractor.

    Args:
        IMG_SIZE (int): The height and width of the input images.

    Returns:
        keras.Model: A feature extractor model.
    """
    # Load the InceptionV3 model pre-trained on ImageNet
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top= False,  # Remove the classification head
        pooling= "avg",      # Global average pooling
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    

    # Define the input and apply preprocessing
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    

    preprocessed = preprocess_input(inputs)

    # Pass the preprocessed input through the feature extractor
    outputs = feature_extractor(preprocessed)
    
    # Build and return the model
    return keras.Model(inputs, outputs, name="feature_extractor")

def CNN_3D(IMG_SIZE=128, num_classes=5):
    input_img = Input(shape=(52, IMG_SIZE, IMG_SIZE, 1), name="image_input")  # define image input tensor
    input_clinical = Input(shape=(2,), name="clinical_input")  # age and gender

    # CNN layers for image input
    x = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(x)
    x = MaxPooling3D((2, 2, 2))(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling3D()(x)
    x = Flatten()(x)

    # Dense layers after concatenation
    combined = Dense(128, activation='relu')(x)
    combined = Concatenate()([combined, input_clinical])
    combined = Dense(32, activation='relu')(combined)

    output = Dense(num_classes, activation='softmax')(combined)

    model = Model(inputs=[input_img, input_clinical], outputs=output)
    model.summary()
    return model

def CNN_RNN(IMG_SIZE: int, time_steps: int, preprocess_input, rnn_units=128, num_classes=5):
    # Your CNN feature extractor (outputs feature vector per image)
    cnn_feature_extractor = CNN_Model(IMG_SIZE, preprocess_input)

    # Input shape: sequence of images (e.g., frames)
    inputs = Input(shape=(52, IMG_SIZE, IMG_SIZE, 3))

    # Apply CNN to each frame using TimeDistributed
    x = TimeDistributed(cnn_feature_extractor)(inputs)  # shape: (batch, time_steps, feature_dim)

    # RNN layer (LSTM or GRU)
    x = LSTM(rnn_units)(x)  # output shape: (batch, rnn_units)

    # Final classification layer
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs, name="CNN_RNN_model")
    return model

def CNN_RNN_Model(MAX_SEQ_LENGTH):
    input_shape = (MAX_SEQ_LENGTH, 128, 128, 3)
    inputs = keras.Input(shape=input_shape)
    
    base_model = InceptionV3(include_top=False, 
                             weights='imagenet',
                             input_shape=(224, 224, 3), 
                             pooling="avg")
    for layer in base_model.layers[:-35]:  # Freeze all but last 2 conv layers
        layer.trainable = False
        
        
    base_model.summary()
    x = keras.layers.TimeDistributed(base_model)(inputs)
    
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.TimeDistributed(Dense(1024, activation='relu'))(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.TimeDistributed(Dense(1024, activation='relu'))(x)
    x = keras.layers.LSTM(256, return_sequences=True)(x)
    x = keras.layers.LSTM(128)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    output = keras.layers.Dense(5, activation="softmax")(x)

    model = keras.Model(inputs, output)
    model.summary()
    return model


def build_rnn_block(frame_features_input):
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, unroll=True,))(frame_features_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128, unroll=True,))(x)
    x = layers.Dropout(0.2)(x)

    return keras.Model(inputs=frame_features_input, outputs=x, name="rnn_block")

def build_his_block_2(inputs, head_size=16, num_heads=1, ff_dim=32, dropout=0.3):
    # LayerNorm + MHA + Residual
    x = LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=head_size, dropout=dropout)(x, x)
    x = Add()([inputs, Dropout(dropout)(attn_output)])  # (None, 2500, 512)

    # Feedforward
    x_ff = LayerNormalization(epsilon=1e-6)(x)
    x_ff = Dense(ff_dim, activation='relu')(x_ff)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(1)(x_ff)  # (None, 2500, 1)

    # Project residual to match shape
    residual = Dense(1)(x)
    x = Add()([residual, x_ff])  # (None, 2500, 1)
    x = Flatten()(x)
    return keras.Model(inputs=inputs, outputs=x, name="his_block")

def build_his_block(his_features_input):
    x = layers.TimeDistributed(Dense(512, activation="relu"))(his_features_input)
    x = layers.TimeDistributed(Dense(128, activation="relu"))(x)
    x = layers.TimeDistributed(Dense(32, activation="relu"))(x)
    x = layers.TimeDistributed(Dense(1, activation="relu"))(x)
    
    # x = Conv1D(512, kernel_size=5, activation='relu', padding='same')(his_features_input)
    # x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    # x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    # x = Conv1D(1, kernel_size=3, activation='relu', padding='same')(x)
    x = Flatten()(x)
    return keras.Model(inputs=his_features_input, outputs=x, name="his_block")


def Multimodel(hp, model_his= None, model_mri = None, ):
    """
    RNN model with clinical input concatenated before the final dense layers.
    
    Args:
        MAX_SEQ_LENGTH (int): Number of image slices/frames.
        NUM_FEATURES (int): Features per image (e.g., from CNN).
        CLINICAL_DIM (int): Number of clinical input features.
    
    Returns:
        keras.Model: The combined RNN model.
    """
    
    # Input 1: Image sequence features from CNN
    his_features_input = keras.Input(shape=(hp["NUM_FRAMES"], hp["FRAME_SIZE"]), name="his_sequence")
    

    # Input 2: Clinical features
    clinical_input = keras.Input(shape=(hp["CLINICAL_DIM"],), name="clinical_features")
    
    
    frame_features_input = keras.Input(shape=(hp["MAX_SEQ_LENGTH"], hp["NUM_FEATURES"]), name="image_sequence")
    
    if not model_his == None and not model_his == None:
        
        rnn_output = model_mri.layers[-7].output 
        mri_block = Model(inputs=model_mri.input, outputs=rnn_output)
        mri = mri_block(frame_features_input)
        
        his_output = model_his.layers[-7].output 
        his_block = Model(inputs=model_his.input, outputs=his_output)
        his = his_block(his_features_input)
        
        for layer in mri_block.layers:
            layer.trainable = False

        # Freeze HIS block
        for layer in his_block.layers:
            layer.trainable = False
    else:
        rnn_block = build_rnn_block(frame_features_input)
        his_block = build_his_block_2(his_features_input)
    
        mri = rnn_block(frame_features_input)
        his = his_block(his_features_input)
        
    x = layers.Dropout(0.2)(his)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Concatenate()([mri, clinical_input])
    # x = layers.Concatenate()([his, clinical_input])
    x = layers.Concatenate()([x, his])
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu", name="dense1")(x)
    x = layers.Dense(128, activation="relu", name="dense2")(x)
    death = layers.Dense(3, activation="softmax", name="death")(x)
    grade = layers.Dense(2, activation="softmax", name="grade")(x)
    mortality = layers.Dense(2, activation="softmax", name="mortality")(x)
    gt = layers.Dense(3, activation="softmax", name="gt")(x)

    # Build the model
    model = keras.Model(inputs=[frame_features_input, his_features_input, clinical_input], outputs=[death, grade, mortality, gt])
    
    # model = keras.Model(inputs=[frame_features_input, his_features_input, clinical_input], outputs=[death, grade])
    model.summary()
    return model



def RNN_Model(MAX_SEQ_LENGTH, NUM_FEATURES, CLINICAL_DIM=2):
    """
    RNN model with clinical input concatenated before the final dense layers.
    
    Args:
        MAX_SEQ_LENGTH (int): Number of image slices/frames.
        NUM_FEATURES (int): Features per image (e.g., from CNN).
        CLINICAL_DIM (int): Number of clinical input features.
    
    Returns:
        keras.Model: The combined RNN model.
    """

    # Input 1: Image sequence features from CNN
    frame_features_input = keras.Input(shape=(MAX_SEQ_LENGTH, NUM_FEATURES), name="image_sequence")

    # Input 2: Clinical features
    clinical_input = keras.Input(shape=(CLINICAL_DIM,), name="clinical_features")

    # RNN part with Bidirectional LSTMs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(frame_features_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(128))(x)
    x = layers.Dropout(0.2)(x)

    # Combine with clinical features
    x = layers.Concatenate()([x, clinical_input])
    x = layers.Dropout(0.2)(x)
    # Final dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
        # Final dense layers
    x = layers.Dense(64, activation="relu")(x)
    output = layers.Dense(5, activation="softmax")(x)

    # Build the model
    model = keras.Model(inputs=[frame_features_input, clinical_input], outputs=output)
    model.summary()
    return model