from keras.models import Model
from keras.layers import Dense, Input, Dropout
from keras.layers import Convolution2D, MaxPooling2D, GlobalAveragePooling2D
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.layers.advanced_activations import ELU
from keras.layers.normalization import BatchNormalization
from classification_models.resnet import ResNet34
from keras.applications.xception import Xception

kernel_initializer = 'he_normal'


def vgg_block(num_filters, block_num, sz=3):
    def f(input_):
        if K.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        b = 'block' + str(block_num)
        x = Convolution2D(num_filters, (sz, sz), kernel_initializer=kernel_initializer, padding='same',
                          name=b + '_conv1')(input_)
        x = ELU(name=b + '_elu1')(x)
        x = BatchNormalization(axis=bn_axis, name=b + '_bn1')(x)
        x = Convolution2D(num_filters, (1, 1), kernel_initializer=kernel_initializer, name=b + '_conv2')(x)
        x = ELU(name=b + '_elu2')(x)
        x = BatchNormalization(axis=bn_axis, name=b + '_bn2')(x)
        x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=b + '_pool')(x)
        return x

    return f


def m46r(include_top=True, input_shape=None, lr=1e-4, loss_balance=0.5, weights=None):
    optimizer = Adam(lr)
    # optimizer = SGD(lr, momentum=0.95, decay=0.0005, nesterov=True)

    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, 3)
        else:
            input_shape = (3, None, None)
    img_input = Input(shape=input_shape)

    # Convolution blocks
    x = vgg_block(32, 1)(img_input)  # Block 1
    x = vgg_block(64, 2)(x)  # Block 2
    x = vgg_block(128, 3)(x)  # Block 3
    x = vgg_block(256, 4)(x)  # Block 4
    x = vgg_block(512, 5)(x)  # Block 5

    x_desc = GlobalAveragePooling2D(name='global_avg_pool')(x)

    if include_top or weights:
        x = Dropout(0.0, name='dropout1')(x_desc)
        x = Dense(1024, kernel_initializer=kernel_initializer, name='fc1')(x)
        x = ELU()(x)
        # x = Dropout(0.3, name='dropout2')(x)
        x_out1 = Dense(1, activation='linear', kernel_initializer=kernel_initializer, name='out1')(x)
        x_out2 = Dense(1, activation='linear', kernel_initializer=kernel_initializer, name='out2')(x)

        model = Model(img_input, outputs=[x_out1, x_out2], name='m46r')

        # load weights
        if weights is not None:
            print('Load weights from', weights)
            model.load_weights(weights)

        model.compile(loss='mse', optimizer=optimizer, metrics=['mape'],
                      loss_weights={'out1': loss_balance, 'out2': 1 - loss_balance})
        return model

    if not include_top:
        model = Model(img_input, x_desc, name='m46r')
        model.compile(loss='mse', optimizer=optimizer, metrics=['mape'])
        return model


def m46c(include_top=True, input_shape=None, channels=2, lr=1e-4, weights=None, classes=4):
    # optimizer = Adam(lr)
    optimizer = SGD(lr, momentum=0.95, decay=0.0005, nesterov=True)

    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, channels)
        else:
            input_shape = (channels, None, None)
    img_input = Input(shape=input_shape)

    # Convolution blocks
    x = vgg_block(32, 1)(img_input)  # Block 1
    x = vgg_block(64, 2)(x)  # Block 2
    x = vgg_block(128, 3)(x)  # Block 3
    x = vgg_block(256, 4)(x)  # Block 4
    x = vgg_block(512, 5)(x)  # Block 5

    x_desc = GlobalAveragePooling2D(name='global_avg_pool')(x)

    if include_top or weights:
        x = Dropout(rate=0.3, name='dropout1')(x_desc)
        x = Dense(512, kernel_initializer=kernel_initializer, name='fc1')(x)
        x = ELU()(x)
        x_out = Dense(classes, activation='softmax', kernel_initializer=kernel_initializer, name='out')(x)

        model = Model(img_input, outputs=[x_out], name='m46c')

        # load weights
        if weights:
            print('Load weights from', weights)
            model.load_weights(weights)

        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        if include_top:
            return model

    if not include_top:
        model = Model(img_input, x_desc, name='m46c')
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model


def r34(input_shape=None, channels=2, lr=1e-4, weights=None, classes=4, **kwargs):

    input_shape_resnet = (None, None, 3) if K.image_data_format() == 'channels_last' else (3, None, None)
    resnet_model = ResNet34(input_shape=input_shape_resnet, include_top=False, weights='imagenet')
    resnet_model = Model(resnet_model.input, resnet_model.get_layer('stage4_unit1_relu1').output)

    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, channels)
        else:
            input_shape = (channels, None, None)
    main_input = Input(input_shape)
    x = Convolution2D(3, (1, 1), kernel_initializer=kernel_initializer)(main_input)
    x = resnet_model(x)
    x = GlobalAveragePooling2D(name='pool1')(x)
    main_output = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(main_input, main_output, name='r34')

    if weights is not None:
        print('Load weights from', weights)
        model.load_weights(weights)

    optimizer = SGD(lr, momentum=0.95, decay=0.0005, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def xcpt(input_shape=None, channels=2, lr=1e-4, weights=None, classes=4, **kwargs):

    input_shape_xception = (None, None, 3) if K.image_data_format() == 'channels_last' else (3, None, None)
    xception_model = Xception(input_shape=input_shape_xception, include_top=False, weights='imagenet')
    xception_model = Model(xception_model.input, xception_model.get_layer('block13_sepconv2_bn').output)

    if input_shape is None:
        if K.image_data_format() == 'channels_last':
            input_shape = (None, None, channels)
        else:
            input_shape = (channels, None, None)
    main_input = Input(input_shape)
    x = Convolution2D(3, (1, 1), kernel_initializer=kernel_initializer)(main_input)
    x = xception_model(x)
    x = GlobalAveragePooling2D(name='pool1')(x)
    main_output = Dense(classes, activation='softmax', name='predictions')(x)

    # Create model.
    model = Model(main_input, main_output, name='xcpt')

    if weights is not None:
        print('Load weights from', weights)
        model.load_weights(weights)

    optimizer = SGD(lr, momentum=0.95, decay=0.0005, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    model = xcpt()
