import torch
import torch.nn as nn
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

class UNet(nn.Module):
    def __init__(self, name="u-net"):
        super(UNet, self).__init__(name=name)
        # Contracting path      
        self.convC11 = nn.Conv2d(1, 64, 3, padding='same')
        nn.init.kaiming_normal_(self.convC11.weight)
        self.reluC11 = nn.ReLU() 
        self.convC12 = nn.Conv2d(64, 64, 3, padding='same')
        nn.init.kaiming_normal_(self.convC12.weight)
        self.bnC12 = nn.BatchNorm2d(64)
        self.reluC12 = nn.ReLU() 
        self.dropC1 = nn.Dropout(0.3)
        self.poolC1 = nn.MaxPool2d(kernel_size=2)

        self.convC21 = nn.Conv2d(64, 128, 3, padding='same')
        nn.init.kaiming_normal_(self.convC21.weight)
        self.reluC21 = nn.ReLU() 
        self.convC22 = nn.Conv2d(128, 128, 3, padding='same')
        nn.init.kaiming_normal_(self.convC22.weight)
        self.bnC22 = nn.BatchNorm2d(128)
        self.reluC22 = nn.ReLU() 
        self.dropC2 = nn.Dropout(0.3)
        self.poolC2 = nn.MaxPool2d(kernel_size=2)

        self.convC31 = nn.Conv2d(128, 256, 3, padding='same')
        nn.init.kaiming_normal_(self.convC31.weight)
        self.reluC31 = nn.ReLU() 
        self.convC32 = nn.Conv2d(256, 256, 3, padding='same')
        nn.init.kaiming_normal_(self.convC32.weight)
        self.bnC32 = nn.BatchNorm2d(256)
        self.reluC32 = nn.ReLU() 
        self.dropC3 = nn.Dropout(0.3)
        self.poolC3 = nn.MaxPool2d(kernel_size=2)

        self.convC41 = nn.Conv2d(256, 512, 3, padding='same')
        nn.init.kaiming_normal_(self.convC41.weight)
        self.reluC41 = nn.ReLU() 
        self.convC42 = nn.Conv2d(512, 512, 3, padding='same')
        nn.init.kaiming_normal_(self.convC42.weight)
        self.bnC42 = nn.BatchNorm2d(512)
        self.reluC42 = nn.ReLU() 
        self.dropC4 = nn.Dropout(0.3)
              
        #Expanding path
        self.upsamp3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconvE3 = nn.Conv2d(512, 512, 2, padding='same')
        nn.init.kaiming_normal_(self.upconvE3.weight)
        self.upreluE3 = nn.ReLU() 
        self.convE31 = nn.Conv2d(1024, 512, 3, padding='same')
        nn.init.kaiming_normal_(self.convE31.weight)
        self.reluE31 = nn.ReLU() 
        self.convE32 = nn.Conv2d(512, 512, 3, padding='same')
        nn.init.kaiming_normal_(self.convE32.weight)
        self.bnE32 = nn.BatchNorm2d(512)
        self.reluE32 = nn.ReLU() 

        self.upsamp2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upconvE2 = nn.Conv2d(512, 512, 2, padding='same')
        nn.init.kaiming_normal_(self.upconvE2.weight)
        self.upreluE2 = nn.ReLU() 
        self.convE21 = nn.Conv2d(1024, 512, 3, padding='same')
        nn.init.kaiming_normal_(self.convE21.weight)
        self.reluE21 = nn.ReLU() 
        self.convE22 = nn.Conv2d(512, 512, 3, padding='same')
        nn.init.kaiming_normal_(self.convE22.weight)
        self.bnE22 = nn.BatchNorm2d(512)
        self.reluE22 = nn.ReLU() 
        
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
    