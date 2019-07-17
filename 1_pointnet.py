import datetime
import logging
import os
import h5py
import numpy as np
import keras
from keras.optimizers import adam
from keras import backend as K
from keras.layers import Input, Dropout, Flatten, Dense, MaxPooling2D, Dot, Lambda, Reshape, BatchNormalization, Activation, Conv1D
from keras.initializers import Constant
from keras.models import Model
from keras.regularizers import Regularizer

class ModelNetProvider(object):

    def __init__(self, h5_path, input_size=None):
        self.input_size = input_size
        self.h5_path = h5_path
        self.x, self.y = self.load_list()

    def load_data(self, path):
        h5 = h5py.File(path, 'r')
        if self.input_size:
            x = h5['data'][:, 0:self.input_size, :]
        else:
            x = h5['data'][()]
        y = h5['label'][()]
        h5.close()
        return x, y

    def load_list(self):
        folder = os.path.dirname(self.h5_path)
        file = open(self.h5_path, 'r')
        x = []
        y = []
        for line in file.readlines():
            path = os.path.join(folder, os.path.basename(line.rstrip('\r\n')))
            x_i, y_i = self.load_data(path)
            if x == [] and y == []:
                x = x_i
                y = y_i
            else:
                x = np.vstack([x, x_i])
                y = np.vstack([y, y_i])
        file.close()
        return x, y

    def generate_samples(self, batch_size, augmentation=False, shuffle=False):
        num_batches = self.x.shape[0] // batch_size
        while True:
            epoch_indices = np.arange(self.x.shape[0])
            if shuffle:
                np.random.shuffle(epoch_indices)
            for i in range(num_batches):
                batch_indices = epoch_indices[0:batch_size]
                epoch_indices = epoch_indices[batch_size:]
                yield self.get_batch(batch_indices, augmentation)
            if epoch_indices.size:
                yield self.get_batch(epoch_indices, augmentation)

    def rotate_point_cloud_by_angle(self, batch_data):
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        return rotated_data

    def jitter_point_cloud(self, batch_data, sigma=0.01, clip=0.05):
        B, N, C = batch_data.shape
        assert (clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
        jittered_data += batch_data
        return jittered_data

    def augment_batch(self, x_batch):
        x_batch = self.rotate_point_cloud_by_angle(x_batch)
        x_batch = self.jitter_point_cloud(x_batch)
        return x_batch

    def get_batch(self, indices, augmentation):
        x_batch = np.copy(self.x[indices])
        y_batch = np.copy(self.y[indices])
        if augmentation:
            x_batch = self.augment_batch(x_batch)
        return x_batch, y_batch

class OrthogonalRegularizer(Regularizer):

    def __init__(self, l1=0.0, l2=0.0):
        self.l1 = K.cast_to_floatx(l1)
        self.l2 = K.cast_to_floatx(l2)

    def __call__(self, x):
        size = int(np.sqrt(x.shape[1].value))
        assert(size * size == x.shape[1].value)
        x = K.reshape(x, (-1, size, size))
        xxt = K.batch_dot(x, x, axes=(2, 2))
        regularization = 0.0
        if self.l1:
            regularization += K.sum(self.l1 * K.abs(xxt - K.eye(size)))
        if self.l2:
            regularization += K.sum(self.l2 * K.square(xxt - K.eye(size)))
        return regularization

    def get_config(self):
        return {'l1': float(self.l1), 'l2': float(self.l2)}

def orthogonal(l1=0.0, l2=0.0):
    return OrthogonalRegularizer(l1=l1, l2=l2)

def dense_bn(x, units, use_bias=True, scope=None, activation=None):
    with K.name_scope(scope):
        x = Dense(units=units, use_bias=use_bias)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(activation)(x)
    return x

def conv1d_bn(x, num_filters, kernel_size, padding='same', strides=1, use_bias=False, scope=None, activation='relu'):
    with K.name_scope(scope):
        input_shape = x.get_shape().as_list()[-2:]
        x = Conv1D(num_filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias, input_shape=input_shape)(x)
        x = BatchNormalization(momentum=0.9)(x)
        x = Activation(activation)(x)
    return x

def transform_net(inputs, scope=None, regularize=False):
    with K.name_scope(scope):
        input_shape = inputs.get_shape().as_list()
        k = input_shape[-1]
        num_points = input_shape[-2]
        net = conv1d_bn(inputs, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='tconv1')
        net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid', use_bias=True, scope='tconv2')
        net = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid', use_bias=True, scope='tconv3')
        net = MaxPooling2D(pool_size=(num_points, 1), padding='valid')(Lambda(K.expand_dims)(net))
        net = Flatten()(net)
        net = dense_bn(net, units=512, scope='tfc1', activation='relu')
        net = dense_bn(net, units=256, scope='tfc2', activation='relu')
        transform = Dense(units=k * k, kernel_initializer='zeros', bias_initializer=Constant(np.eye(k).flatten()), activity_regularizer=orthogonal(l2=0.001) if regularize else None)(net)
        transform = Reshape((k, k))(transform)
    return transform

def pointnet_base(inputs):
    ptransform = transform_net(inputs, scope='transform_net1', regularize=False)
    point_cloud_transformed = Dot(axes=(2, 1))([inputs, ptransform])
    net = conv1d_bn(point_cloud_transformed, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv1')
    net = conv1d_bn(net, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv2')
    ftransform = transform_net(net, scope='transform_net2', regularize=True)
    net_transformed = Dot(axes=(2, 1))([net, ftransform])
    net = conv1d_bn(net_transformed, num_filters=64, kernel_size=1, padding='valid', use_bias=True, scope='conv3')
    net = conv1d_bn(net, num_filters=128, kernel_size=1, padding='valid', use_bias=True, scope='conv4')
    net = conv1d_bn(net, num_filters=1024, kernel_size=1, padding='valid', use_bias=True, scope='conv5')
    return net

def pointnet_cls(input_shape, classes, activation=None):
    assert K.image_data_format() == 'channels_last'
    num_point = input_shape[0]
    inputs = Input(input_shape, name='Input_cloud')
    net = pointnet_base(inputs)
    net = MaxPooling2D(pool_size=(num_point, 1), padding='valid', name='maxpool')(Lambda(K.expand_dims)(net))
    net = Flatten()(net)
    if isinstance(classes, dict):
        net = [dense_bn(net, units=512, scope=r + '_fc1', activation='relu') for r in classes]
        net = [Dropout(0.3, name=r + '_dp1')(n) for r, n in zip(classes, net)]
        net = [dense_bn(n, units=256, scope=r + '_fc2', activation='relu') for r, n in zip(classes, net)]
        net = [Dropout(0.3, name=r + '_dp2')(n) for r, n in zip(classes, net)]
        net = [Dense(units=classes[r], activation=activation, name=r)(n) for r, n in zip(classes, net)]
    else:
        net = dense_bn(net, units=512, scope='fc1', activation='relu')
        net = Dropout(0.3, name='dp1')(net)
        net = dense_bn(net, units=256, scope='fc2', activation='relu')
        net = Dropout(0.3, name='dp2')(net)
        net = Dense(units=classes, name='fc3', activation=activation)(net)
    model = Model(inputs, net, name='pointnet_cls')
    return model

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    train_list = '2_dataset/train_files.txt'
    test_list = '2_dataset/test_files.txt'
    weights_path = os.path.expanduser('/home/dgxuser103/team4/3_weights.h5')
    log_dir = os.path.expanduser('/home/dgxuser103/team4/4_log')
    log_dir = os.path.expanduser(log_dir)
    folder_name = 'pointnet_modelnet_' + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    log_dir = os.path.join(log_dir, folder_name)
    os.makedirs(log_dir)

    epochs = 40
    input_size = 2048
    batch_size = 32
    num_classes = 40

    model = pointnet_cls((input_size, 3), classes=num_classes, activation='softmax')
    loss = 'sparse_categorical_crossentropy'
    metric = ['sparse_categorical_accuracy']
    monitor = 'val_loss'
    callbacks = list()
    callbacks.append(keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True))
    callbacks.append(keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=3, verbose=1, min_lr=1e-10))
    callbacks.append(keras.callbacks.EarlyStopping(monitor=monitor, patience=10))
    callbacks.append(keras.callbacks.ModelCheckpoint(weights_path, monitor=monitor, verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    train_dataset = ModelNetProvider(train_list, input_size=input_size)
    train_generator = train_dataset.generate_samples(batch_size=batch_size, augmentation=True, shuffle=True)
    train_steps_per_epoch = (train_dataset.x.shape[0] // batch_size) + 1
    val_dataset = ModelNetProvider(test_list, input_size=input_size)
    val_generator = val_dataset.generate_samples(batch_size=batch_size, augmentation=False)
    val_steps_per_epoch = (val_dataset.x.shape[0] // batch_size) + 1
    optimizer = adam(lr=1e-3)
    model.compile(loss=loss, optimizer=optimizer, metrics=metric)
    history = model.fit_generator(train_generator, train_steps_per_epoch, validation_data=val_generator, validation_steps=val_steps_per_epoch, epochs=epochs, callbacks=callbacks)