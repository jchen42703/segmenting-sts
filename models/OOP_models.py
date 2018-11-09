from keras.models import Model
from keras.layers import Input
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D

from keras.callbacks import EarlyStopping
# from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import keras.backend as K

class Unet2d(object):
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
    
class DeepLab(object)
    ''' Inspired by: https://github.com/bonlime/keras-deeplab-v3-plus'''
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
             
    @staticmethod
    def build_model(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2', OS=16, alpha=1.):
        """ Instantiates the Deeplabv3+ architecture
        Optionally loads weights pre-trained
        on PASCAL VOC. This model is available for TensorFlow only,
        and can only be used with inputs following the TensorFlow
        data format `(width, height, channels)`.
        # Arguments
            weights: one of 'pascal_voc' (pre-trained on pascal voc)
                or None (random initialization)
            input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
                to use as image input for the model.
            input_shape: shape of input image. format HxWxC
                PASCAL VOC model was trained on (512,512,3) images
            classes: number of desired classes. If classes != 21,
                last layer is initialized randomly
            backbone: backbone to use. one of {'xception','mobilenetv2'}
            OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
                Used only for xception backbone.
            alpha: controls the width of the MobileNetV2 network. This is known as the
                width multiplier in the MobileNetV2 paper.
                    - If `alpha` < 1.0, proportionally decreases the number
                        of filters in each layer.
                    - If `alpha` > 1.0, proportionally increases the number
                        of filters in each layer.
                    - If `alpha` = 1, default number of filters from the paper
                        are used at each layer.
                Used only for mobilenetv2 backbone
        # Returns
            A Keras model instance.
        # Raises
            RuntimeError: If attempting to run this model with a
                backend that does not support separable convolutions.
            ValueError: in case of invalid argument for `weights` or `backbone`
        """

        if not (weights in {'pascal_voc', None}):
            raise ValueError('The `weights` argument should be either '
                             '`None` (random initialization) or `pascal_voc` '
                             '(pre-trained on PASCAL VOC)')

        if K.backend() != 'tensorflow':
            raise RuntimeError('The Deeplabv3+ model is only available with '
                               'the TensorFlow backend.')

        if not (backbone in {'xception', 'mobilenetv2'}):
            raise ValueError('The `backbone` argument should be either '
                             '`xception`  or `mobilenetv2` ')

        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        if backbone == 'xception':
            if OS == 8:
                entry_block3_stride = 1
                middle_block_rate = 2  # ! Not mentioned in paper, but required
                exit_block_rates = (2, 4)
                atrous_rates = (12, 24, 36)
            else:
                entry_block3_stride = 2
                middle_block_rate = 1
                exit_block_rates = (1, 2)
                atrous_rates = (6, 12, 18)

            x = Conv2D(32, (3, 3), strides=(2, 2),
                       name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
            x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
            x = Activation('relu')(x)

            x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
            x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
            x = Activation('relu')(x)

            x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                                skip_connection_type='conv', stride=2,
                                depth_activation=False)
            x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                                       skip_connection_type='conv', stride=2,
                                       depth_activation=False, return_skip=True)

            x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                                skip_connection_type='conv', stride=entry_block3_stride,
                                depth_activation=False)
            for i in range(16):
                x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                                    skip_connection_type='sum', stride=1, rate=middle_block_rate,
                                    depth_activation=False)

            x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                                skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
                                depth_activation=False)
            x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                                skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                                depth_activation=True)

        else:
            OS = 8
            first_block_filters = _make_divisible(32 * alpha, 8)
            x = Conv2D(first_block_filters,
                       kernel_size=3,
                       strides=(2, 2), padding='same',
                       use_bias=False, name='Conv')(img_input)
            x = BatchNormalization(
                epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
            x = Activation(relu6, name='Conv_Relu6')(x)

            x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                    expansion=1, block_id=0, skip_connection=False)

            x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                    expansion=6, block_id=1, skip_connection=False)
            x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                    expansion=6, block_id=2, skip_connection=True)

            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                    expansion=6, block_id=3, skip_connection=False)
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                    expansion=6, block_id=4, skip_connection=True)
            x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                    expansion=6, block_id=5, skip_connection=True)

            # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
                                    expansion=6, block_id=6, skip_connection=False)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=7, skip_connection=True)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=8, skip_connection=True)
            x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=9, skip_connection=True)

            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=10, skip_connection=False)
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=11, skip_connection=True)
            x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
                                    expansion=6, block_id=12, skip_connection=True)

            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
                                    expansion=6, block_id=13, skip_connection=False)
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                    expansion=6, block_id=14, skip_connection=True)
            x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
                                    expansion=6, block_id=15, skip_connection=True)

            x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
                                    expansion=6, block_id=16, skip_connection=False)

        # end of feature extractor

        # branching for Atrous Spatial Pyramid Pooling

        # Image Feature branch
        #out_shape = int(np.ceil(input_shape[0] / OS))
        b4 = AveragePooling2D(pool_size=(int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(x)
        b4 = Conv2D(256, (1, 1), padding='same',
                    use_bias=False, name='image_pooling')(b4)
        b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
        b4 = Activation('relu')(b4)
        b4 = BilinearUpsampling((int(np.ceil(input_shape[0] / OS)), int(np.ceil(input_shape[1] / OS))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
        b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
        b0 = Activation('relu', name='aspp0_activation')(b0)

        # there are only 2 branches in mobilenetV2. not sure why
        if backbone == 'xception':
            # rate = 6 (12)
            b1 = SepConv_BN(x, 256, 'aspp1',
                            rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
            # rate = 12 (24)
            b2 = SepConv_BN(x, 256, 'aspp2',
                            rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
            # rate = 18 (36)
            b3 = SepConv_BN(x, 256, 'aspp3',
                            rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)

            # concatenate ASPP branches & project
            x = Concatenate()([b4, b0, b1, b2, b3])
        else:
            x = Concatenate()([b4, b0])

        x = Conv2D(256, (1, 1), padding='same',
                   use_bias=False, name='concat_projection')(x)
        x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
        x = Activation('relu')(x)
        x = Dropout(0.1)(x)

        # DeepLab v.3+ decoder

        if backbone == 'xception':
            # Feature projection
            # x4 (x2) block
            x = BilinearUpsampling(output_size=(int(np.ceil(input_shape[0] / 4)),
                                                int(np.ceil(input_shape[1] / 4))))(x)
            dec_skip1 = Conv2D(48, (1, 1), padding='same',
                               use_bias=False, name='feature_projection0')(skip1)
            dec_skip1 = BatchNormalization(
                name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
            dec_skip1 = Activation('relu')(dec_skip1)
            x = Concatenate()([x, dec_skip1])
            x = SepConv_BN(x, 256, 'decoder_conv0',
                           depth_activation=True, epsilon=1e-5)
            x = SepConv_BN(x, 256, 'decoder_conv1',
                           depth_activation=True, epsilon=1e-5)

        # you can use it with arbitary number of classes
        if classes == 21:
            last_layer_name = 'logits_semantic'
        else:
            last_layer_name = 'custom_logits_semantic'

        x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
        x = BilinearUpsampling(output_size=(input_shape[0], input_shape[1]))(x)

        # Ensure that the model takes into account
        # any potential predecessors of `input_tensor`.
        if input_tensor is not None:
            inputs = get_source_inputs(input_tensor)
        else:
            inputs = img_input

        model = Model(inputs, x, name='deeplabv3plus')

        # load weights

        if weights == 'pascal_voc':
            if backbone == 'xception':
                weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH_X,
                                        cache_subdir='models')
            else:
                weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
                                        WEIGHTS_PATH_MOBILE,
                                        cache_subdir='models')
            model.load_weights(weights_path, by_name=True)
        return model
            model = Model(input=model_in, output=model_out)

            return model
