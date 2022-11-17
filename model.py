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

class UNet(tf.keras.Model):
    def __init__(self, input_size, name="u-net"):
        super(UNet, self).__init__(name=name)
        
        self.conv11 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv12 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.batch1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation('relu')
        self.pool1 = tf.keras.layers.MaxPool2D()

        self.conv21 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv22 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.batch2 = tf.keras.layers.BatchNormalization() 
        self.act2 = tf.keras.layers.Activation('relu')
        self.drop2 = tf.keras.layers.Dropout(0.3)
        self.pool2 = tf.keras.layers.MaxPool2D()
        
        self.conv31 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv32 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.batch3 = tf.keras.layers.BatchNormalization() 
        self.act3 = tf.keras.layers.Activation('relu')
        self.drop3 = tf.keras.layers.Dropout(0.3)

        #Expanding path
        self.upsamp4 = tf.keras.layers.UpSampling2D()
        self.conv4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.batch4 = tf.keras.layers.BatchNormalization() 
        self.act4 = tf.keras.layers.Activation('relu')

        self.upsamp5 = tf.keras.layers.UpSampling2D()
        self.conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv5 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.output = tf.keras.layers.Conv2D(1, 1, activation='sigmoid')
    
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch1(x)
        x_cat5 = self.act1(x)
        x = self.pool1(x_cat5)

        x = self.conv2(x)
        x = self.batch2(x)
        x_cat4 = self.act2(x)
        x = self.drop2(x_cat4)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch3(x)
        x = self.act3(x)
        x = self.drop3(x)

        x = self.upsamp4(x)
        x = tf.concat([x_cat4, x], axis=3)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.act4(x)

        x = self.upsamp5(x)
        x = tf.concat([x_cat5, x], axis=3)
        x = self.conv4(x)
        x = self.batch4(x)
        x = self.act4(x)
        
        return x        