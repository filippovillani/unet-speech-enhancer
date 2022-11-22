import tensorflow as tf

def unet(input_size=(96,248,1)):
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

class UNet(tf.keras.layers.Layer):
    def __init__(self, name="u-net"):
        super(UNet, self).__init__(name=name)
        # Contracting path
        self.convC11 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convC11')
        self.convC12 = tf.keras.layers.Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal', name='convC12')
        self.batchC1 = tf.keras.layers.BatchNormalization(name='batchC1')
        self.actC1 = tf.keras.layers.Activation('relu', name='actC1')
        self.dropC1 = tf.keras.layers.Dropout(0.3, name='actC1')
        self.poolC1 = tf.keras.layers.MaxPool2D(name='actC1')

        self.convC21 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convC21')
        self.convC22 = tf.keras.layers.Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal', name='convC22')
        self.batchC2 = tf.keras.layers.BatchNormalization(name='batchC2') 
        self.actC2 = tf.keras.layers.Activation('relu', name='actC2')
        self.dropC2 = tf.keras.layers.Dropout(0.3, name='dropC2')
        self.poolC2 = tf.keras.layers.MaxPool2D(name='poolC2')

        self.convC31 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convC31')
        self.convC32 = tf.keras.layers.Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal', name='convC32')
        self.batchC3 = tf.keras.layers.BatchNormalization(name='batchC3') 
        self.actC3 = tf.keras.layers.Activation('relu', name='actC3')
        self.dropC3 = tf.keras.layers.Dropout(0.3, name='dropC3')
        self.poolC3 = tf.keras.layers.MaxPool2D(name='poolC3')
        
        self.convC41 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convC41')
        self.convC42 = tf.keras.layers.Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal', name='convC42')
        self.dropC4 = tf.keras.layers.Dropout(0.3, name='dropC4')
        
        #Expanding path
        self.upsamp3 = tf.keras.layers.UpSampling2D(name='upsamp3')
        self.upconvE3 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='upconvE3')
        self.convE31 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convE31')
        self.convE32 = tf.keras.layers.Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal', name='convE32')
        self.batchE3 = tf.keras.layers.BatchNormalization(name='batchE3') 
        self.actE3 = tf.keras.layers.Activation('relu', name='actE3')

        self.upsamp2 = tf.keras.layers.UpSampling2D(name='upsamp2')
        self.upconvE2 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='upconvE2')
        self.convE21 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convE21')
        self.convE22 = tf.keras.layers.Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal', name='convE22')
        self.batchE2 = tf.keras.layers.BatchNormalization(name='batchE2') 
        self.actE2 = tf.keras.layers.Activation('relu', name='actE2')
         
        self.upsamp1 = tf.keras.layers.UpSampling2D(name='upsamp1')
        self.upconvE1 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal', name='upconvE1')
        self.convE11 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convE11')
        self.convE12 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convE12')
        self.convE13 = tf.keras.layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='convE13')
        self.out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='out')
    
    def call(self, inputs):
        x = self.convC11(inputs)
        x = self.convC12(x)
        x = self.batchC1(x)
        x_cat1 = self.actC1(x)
        x = self.dropC1(x_cat1)
        x = self.poolC1(x)

        x = self.convC21(x)
        x = self.convC22(x)
        x = self.batchC2(x)
        x_cat2 = self.actC2(x)
        x = self.dropC2(x_cat2)
        x = self.poolC2(x)
        
        x = self.convC31(x)
        x = self.convC32(x)
        x = self.batchC3(x)
        x_cat3 = self.actC3(x)
        x = self.dropC3(x_cat3)
        x = self.poolC3(x)
        
        x = self.convC41(x)
        x = self.convC42(x)
        x = self.dropC4(x)

        x = self.upsamp3(x)
        x = self.upconvE3(x)
        x = tf.concat([x_cat3, x], axis=3)
        x = self.convE31(x)
        x = self.convE32(x)
        x = self.batchE3(x)
        x = self.actE3(x)

        x = self.upsamp2(x)
        x = self.upconvE2(x)
        x = tf.concat([x_cat2, x], axis=3)
        x = self.convE21(x)
        x = self.convE22(x)
        x = self.batchE2(x)
        x = self.actE2(x)
        
        x = self.upsamp1(x)
        x = self.upconvE1(x)
        x = tf.concat([x_cat1, x], axis=3)
        x = self.convE11(x)
        x = self.convE12(x)
        x = self.convE13(x)
        x = self.out(x)
        
        return x    
    
    def build_model(input_size):
        inputs = tf.keras.layers.Input(input_size)
        model = UNet()
        outputs = model(inputs)              
        model = tf.keras.Model(inputs = inputs, outputs = outputs)
        return model    
    