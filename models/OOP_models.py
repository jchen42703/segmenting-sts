from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras.callbacks import EarlyStopping
# from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K

class unet_2d(object):
    '''
        2D U-Net Keras implementation

    methods:
    * model: returns a keras.models.Model
    * compile_model: returns a compiled model with a specified learning rate, lr (dice loss, amsgrad)
    * get_possible_input_shapes: returns a list of possible image shapes compatible with the model
    '''
    def __init__(self, n_fea = [64, 128, 256, 512, 1024], patch_size = (96,96,1), num_classes=2,
                 conv_kernel = (3,3), deconv_kernel = (2,2), activation = 'relu', padding = 'same',
                 ):
        '''
        n_fea: list of number of filters for each stack of convs; currently, only supports 5
        patch_size: window size; (h,w,1)
        num_classes: int
        conv_kernel: 2D
        deconv_kernel: 2D
        activation: 
        padding:
        '''
        self.n_fea = n_fea
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.conv_kernel = conv_kernel
        self.deconv_kernel = deconv_kernel
        self.padding = padding
        self.activation = activation
                     
    def model(self): 
        '''
        returns functional model for keras
        '''
        input_shape = self.patch_size
        
        inputs = Input(input_shape)
        conv1 = Conv2D(self.n_fea[0], (3, 3), activation=self.activation, padding=self.padding)(inputs)
        conv1 = Conv2D(self.n_fea[0], (3, 3), activation=self.activation, padding=self.padding)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(self.n_fea[1], (3, 3), activation=self.activation, padding=self.padding)(pool1)
        conv2 = Conv2D(self.n_fea[1], (3, 3), activation=self.activation, padding=self.padding)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(self.n_fea[2], (3, 3), activation=self.activation, padding=self.padding)(pool2)
        conv3 = Conv2D(self.n_fea[2], (3, 3), activation=self.activation, padding=self.padding)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(self.n_fea[3], (3, 3), activation=self.activation, padding=self.padding)(pool3)
        conv4 = Conv2D(self.n_fea[3], (3, 3), activation=self.activation, padding=self.padding)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        
        conv5 = Conv2D(self.n_fea[4], (3, 3), activation=self.activation, padding=self.padding)(pool4)
        conv5 = Conv2D(self.n_fea[4], (3, 3), activation=self.activation, padding=self.padding)(conv5)
        
        up4 = concatenate([Conv2DTranspose(self.n_fea[3], (2, 2), strides=(2, 2), padding=self.padding)(conv5), conv4], 
                          axis=3)
        conv6 = Conv2D(self.n_fea[3], (3, 3), activation=self.activation, padding=self.padding)(up4)
        conv6 = Conv2D(self.n_fea[3], (3, 3), activation=self.activation, padding=self.padding)(conv6)

        up3 = concatenate([Conv2DTranspose(self.n_fea[2], (2, 2), strides=(2, 2), padding=self.padding)(conv6), conv3], 
                          axis=3)
        conv7 = Conv2D(self.n_fea[2], (3, 3), activation=self.activation, padding=self.padding)(up3)
        conv7 = Conv2D(self.n_fea[2], (3, 3), activation=self.activation, padding=self.padding)(conv7)
        
        up2 = concatenate([Conv2DTranspose(self.n_fea[1], (2, 2), strides=(2, 2), padding=self.padding)(conv7), conv2], 
                          axis=3)
        conv8 = Conv2D(self.n_fea[1], (3, 3), activation=self.activation, padding=self.padding)(up2)
        conv8 = Conv2D(self.n_fea[1], (3, 3), activation=self.activation, padding=self.padding)(conv8)
        
        up1 = concatenate([Conv2DTranspose(self.n_fea[0], (2, 2), strides=(2, 2), padding=self.padding)(conv8), conv1], 
                          axis=3)
        conv9 = Conv2D(self.n_fea[0], (3, 3), activation=self.activation, padding=self.padding)(up1)
        conv9 = Conv2D(self.n_fea[0], (3, 3), activation=self.activation, padding=self.padding)(conv9)
        
        if self.num_classes == 2:
            classifier = Conv2D(1, (1,1), activation = 'sigmoid')(conv9) # FOR BINARY SEGMENTATION ONLY
        else: 
            classifier = Conv2D(self.num_classes, (1,1), activation = 'softmax')(conv9)

        model = Model(inputs=[inputs], outputs=[classifier])
        # need to do argmax for post-processing
        return model
            
    def compile_model(self, lr): 
        '''
        compiles model
        lr: learning rate
        '''
        # compiling model
        #defining loss/evaluation functions
        if self.num_classes == 2: 
            def dice_coef(y_true, y_pred, smooth=1):
                  # for binary only 
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(y_pred)
                intersection = K.sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
                              
        else: 
            def dice_coef(y_true, y_pred, smooth = 1):
                y_true_f = K.flatten(y_true)
                y_pred_f = K.flatten(K.max(y_pred, axis = -1))
                intersection = K.sum(y_true_f * y_pred_f)
                return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
# doesn't work bc not differentiable
#             y_true_f = K.flatten(y_true)
#             y_pred_f = K.flatten(K.cast(K.argmax(y_pred, axis = -1), 'float32'))
#             intersection = K.sum(y_true_f * y_pred_f)
#             return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

        def dice_coef_loss(y_true, y_pred):
            return -dice_coef(y_true, y_pred)
        
        model = unet_2d(n_fea = self.n_fea, patch_size = self.patch_size, num_classes=self.num_classes,
                conv_kernel = self.conv_kernel, deconv_kernel = self.deconv_kernel, 
                padding = self.padding).model()
        
        #optimizer = SGD(lr=lr, momentum=0.0, decay=0.0, nesterov=False)
        optimizer = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
        model.compile(optimizer = optimizer, loss = dice_coef_loss, metrics = [dice_coef])
        
        return model
               
    def get_possible_input_shapes(self):
        '''
        returns a list of possible cropped input shapes for your desired data 
        '''
        image_shapes = []
        for shape in range(0,self.patch_size[0]+1): 
            try: 
                network = unet_2d(patch_size = (shape,shape,1), padding = self.padding)
                network.model()
                image_shapes.append(network.patch_size)
            except ValueError: 
                pass
        return image_shapes
    
class DilatedCNN(object)
    ''' Multi-Scale Dilated CNN'''
    def __init__(self, input_shape = (56, 56, 1), activation = 'relu', num_classes = 2):
        self.input_shape = input_shape
        self.activation = activation
        self.num_classes = num_classes
    
#     @staticmethod
    def conv_block(self, input, n_filters, kernel_size = (3,3), activation = 'relu', pool = True):
        conv = Convolution2D(64, kernel_size, activation = activation)(input)
        conv = Convolution2D(64, kernel_size, activation = activation)(conv)
        conv = Convolution2D(64, (3,3), activation = activation)(conv)
        if pool: 
            conv = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
        return conv
             
    def build_model(self):
        '''
        deprecated syntax
        '''
        model_in = Input(input_shape)
        # downsampling
        pool1 = self.conv_block(model_in, n_filters = 64, kernel_size = (3,3), activation = 'relu')
        pool2 = self.conv_block(pool1, n_filters = 128, kernel_size = (3,3), activation = 'relu')
        pool3 = self.conv_block(pool2, n_filters = 256, kernel_size = (3,3), activation = 'relu')
        conv4 = self.conv_block(pool3, n_filters = 512, kernel_size = (3,3), activation = 'relu', pool = False)
        
        # atrous convs
        h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_1')(h)
        h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_2')(h)
        h = AtrousConvolution2D(512, 3, 3, atrous_rate=(2, 2), activation='relu', name='conv5_3')(h)
        h = AtrousConvolution2D(4096, 7, 7, atrous_rate=(4, 4), activation='relu', name='fc6')(h)
        
        
        h = Dropout(0.5, name='drop6')(h)
        h = Convolution2D(4096, 1, 1, activation='relu', name='fc7')(h)
        h = Dropout(0.5, name='drop7')(h)
        h = Convolution2D(classes, 1, 1, name='final')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_1')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Convolution2D(classes, 3, 3, activation='relu', name='ctx_conv1_2')(h)
        h = ZeroPadding2D(padding=(2, 2))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(2, 2), activation='relu', name='ctx_conv2_1')(h)
        h = ZeroPadding2D(padding=(4, 4))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(4, 4), activation='relu', name='ctx_conv3_1')(h)
        h = ZeroPadding2D(padding=(8, 8))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(8, 8), activation='relu', name='ctx_conv4_1')(h)
        h = ZeroPadding2D(padding=(16, 16))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(16, 16), activation='relu', name='ctx_conv5_1')(h)
        h = ZeroPadding2D(padding=(32, 32))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(32, 32), activation=self.activation, name='ctx_conv6_1')(h)
        h = ZeroPadding2D(padding=(64, 64))(h)
        h = AtrousConvolution2D(classes, 3, 3, atrous_rate=(64, 64), activation=self.activation, name='ctx_conv7_1')(h)
        h = ZeroPadding2D(padding=(1, 1))(h)
        h = Convolution2D(self.num_classes, 3, 3, activation='relu', name='ctx_fc1')(h)
        h = Convolution2D(self.num_classes, 1, 1, name='ctx_final')(h)

        # the following two layers pretend to be a Deconvolution with grouping layer.
        # never managed to implement it in Keras
        # since it's just a gaussian upsampling trainable=False is recommended
        h = UpSampling2D(size=(8, 8))(h)
        logits = Convolution2D(self.num_classes, 16, 16, border_mode='same', bias=False, trainable=False, name='ctx_upsample')(h)

        if apply_softmax:
            model_out = softmax(logits)
        else:
            model_out = logits

        model = Model(input=model_in, output=model_out)

        return model
