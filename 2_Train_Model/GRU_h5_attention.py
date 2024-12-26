import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, Layer
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, TensorBoard, ModelCheckpoint
import os
import numpy as np
from datetime import datetime
import gc   
import pandas as pd
import h5py
import itertools


my_batch_size = 8 
my_learning_rate = 0.001
Input_n          = 16
Output_n         = 750 
cpu = 16
first_layer = 256
second_layer = 128
saveEpochs = 2 # after 2 saved


path_train = "/mnt/researchers/claudia-prieto/datasets/GRU_DATASET/train/"
path_valid = "/mnt/researchers/claudia-prieto/datasets/GRU_DATASET/valid/"
folder_name = "Models_GRU_att_BS_8"


checkpoint_path = folder_name + "/checkpoints/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, dataset_dir, batch_size=32, dimx=(32,32,32),
                dimy=(32,32,32), n_channels=1, shuffle=True):
        'Initialization'
        self.dimx = dimx
        self.dimy = dimy
        self.batch_size = batch_size
        self.dataset_dir = dataset_dir
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.parameters_file = 'parameters.h5'
        self.fingerprints_file = 'fingerprints.h5'
        with h5py.File(self.dataset_dir + self.parameters_file, 'r') as f:
            self.n_samples = len(f['data'])
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_samples / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle: 
            np.random.shuffle(self.indexes)
        gc.collect()

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        X = self._read_h5_data(self.parameters_file, list_IDs_temp)
        y = self._read_h5_data(self.fingerprints_file, list_IDs_temp)
        X = X.reshape((self.batch_size, *self.dimx))
        y = y.reshape((self.batch_size, *self.dimy))
        return X, y

    def _read_h5_data(self, filename, indexes):
        'Reads data from HDF5 file for specified indexes'
        with h5py.File(self.dataset_dir + filename, 'r') as h5f:
            data = h5f['data'][sorted(indexes)]
        return data


class CustomAttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomAttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(14, 1),
                                initializer='random_normal',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[1], 1),
                                initializer='zeros',
                                trainable=True)
        super(CustomAttentionLayer, self).build(input_shape)

    def call(self, x):
        # Extract PP-interval 
        pp_intervals = x[:, :, -14:]
        e = tf.keras.backend.tanh(tf.keras.backend.dot(pp_intervals, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e, axis=1)
        context = x * tf.keras.backend.repeat_elements(alpha, x.shape[2], axis=2)
        context = tf.keras.backend.sum(context, axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


model = Sequential([
    GRU(first_layer, return_sequences=True, input_shape=(Input_n, 1)),
    GRU(second_layer, return_sequences=True),
    CustomAttentionLayer(),
    Dense(Output_n)
])



params = {'batch_size': my_batch_size, 
        'dimx': (Input_n, 1),  
        'dimy': (Output_n,),    
        'n_channels': 1,
        'shuffle': True}

training_generator   = DataGenerator(path_train, **params)
validation_generator = DataGenerator(path_valid, **params)
 

def step_decay(epoch): 
    initial_lrate = my_learning_rate
    drop          = 0.5
    epochs_drop   = 10.0
    lrate         = initial_lrate * drop ** np.floor((1+epoch)/epochs_drop)
    return lrate

# Train the model
lrate                = LearningRateScheduler(step_decay, verbose=1)
monitor              = EarlyStopping(monitor='val_mean_absolute_error', patience=13, min_delta=0.00001, verbose=0, restore_best_weights=True, mode='min')
tensorboard_callback = TensorBoard(log_dir="./logs")
 

# callback that saves the model's weights every 3 epochs
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    period=saveEpochs)


# Check if there are model weights 
latest = tf.train.latest_checkpoint(checkpoint_dir)
if latest:
    print(f"Resuming training from: {latest}")
    model.load_weights(latest)



# Compile the model
#model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate), loss='mse')
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=my_learning_rate), loss=tf.keras.losses.LogCosh(), metrics=[tf.keras.metrics.MeanAbsoluteError()]) # not so harsh against outliers
history = model.fit(x=training_generator, 
                    validation_data     = validation_generator,
                    epochs              = 100,
                    workers             = cpu,
                    use_multiprocessing = True,
                    verbose             = 1,
                    callbacks           = [lrate, monitor, tensorboard_callback, cp_callback])



print ('Saving the model now')
current_datetime = datetime.now()
#timestamp_string = current_datetime.strftime("%y-%m-%d__%H-%M") 
timestamp_string = current_datetime.strftime("%m-%d")
lrSci            = '{:.0E}'.format(my_learning_rate)
model_directory  = f"GRU_Full_{timestamp_string}_LR_{lrSci}_BS_{my_batch_size}_att_pp"
model_path       = os.path.join(folder_name, model_directory)
os.makedirs(model_path, exist_ok=True)  

model.save(model_path)  # Save the entire model

history_filename = f"training_history.npy"
history_path     = os.path.join(model_path, history_filename)
np.save(history_path, history.history)















