import tensorflow as tf

def unet(input_size):
    inputs = tf.keras.layers.Input(input_size)
    
    # Contracting path
    conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs)
    batch1 = tf.keras.layers.BatchNormalization()(conv1)
    act1 = tf.keras.layers.Activation('relu')(batch1)
    pool1 = tf.keras.layers.MaxPool2D()(act1) # default size is (2,2)

    conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(pool1)
    batch2 = tf.keras.layers.BatchNormalization()(conv2) 
    act2 = tf.keras.layers.Activation('relu')(batch2)
    drop2 = tf.keras.layers.Dropout(0.3)(act2)
    pool2 = tf.keras.layers.MaxPool2D()(drop2)
    
    conv3 = tf.keras.layers.Conv2D(256, 3, padding='same', kernel_initializer='he_normal')(pool2)
    batch3 = tf.keras.layers.BatchNormalization()(conv3) 
    act3 = tf.keras.layers.Activation('relu')(batch3)
    drop3 = tf.keras.layers.Dropout(0.3)(act3)

    #Expanding path
    upsamp4 = tf.keras.layers.UpSampling2D()(drop3) # default size is (2,2)
    concat4 = tf.concat([act2, upsamp4], axis=3)
    conv4 = tf.keras.layers.Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(concat4)
    batch4 = tf.keras.layers.BatchNormalization()(conv4) 
    act4 = tf.keras.layers.Activation('relu')(batch4)


    upsamp5 = tf.keras.layers.UpSampling2D()(act4)
    concat5 = tf.concat([act1, upsamp5], axis=3)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat5)
    conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')(conv5)

    model = tf.keras.Model(inputs = inputs, outputs = output)
    return model
