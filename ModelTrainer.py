import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from sklearn.utils import shuffle
from tensorflow.python.framework import ops

from ImagesLoader import *


class ModelTrainer:

    def train_network(self):
        # Train network,
        self.model.fit(self.loadedImages, self.outputVectors, n_epoch=20,
                       validation_set=(
                           load_test_images(),
                           init_test_labels()
                       ),
                       snapshot_step=100, show_metric=True, run_id='conv_net_coursera')  #
        self.model.save("TrainedModel/GestureRecogModel.tfl")

    def cnn(self):
        self.conv_net = input_data(shape=[None, 89, 100, 1], name='input')
        # 32 -> 64
        self.conv_net = conv_2d(self.conv_net, 32, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)
        # 64 -> 128
        self.conv_net = conv_2d(self.conv_net, 64, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)
        # 128 -> 256
        self.conv_net = conv_2d(self.conv_net, 128, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)
        # 256 -> 256
        self.conv_net = conv_2d(self.conv_net, 256, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)
        # 256 -> 128
        self.conv_net = conv_2d(self.conv_net, 256, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)
        # 128 -> 64
        self.conv_net = conv_2d(self.conv_net, 128, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)
        # 64
        self.conv_net = conv_2d(self.conv_net, 64, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)

        self.conv_net = fully_connected(self.conv_net, 1000, activation='relu')
        self.conv_net = dropout(self.conv_net, 0.75)
        # 64 -> 8
        # 8 neurons on the output
        self.conv_net = fully_connected(self.conv_net, 8, activation='softmax')

        self.conv_net = regression(self.conv_net, optimizer='adam', learning_rate=0.001,
                                   loss='categorical_crossentropy',
                                   name='regression')
        self.model = tflearn.DNN(self.conv_net, tensorboard_verbose=0)

    def __init__(self):
        ops.reset_default_graph()
        # Shuffle data
        self.loadedImages, self.outputVectors = shuffle(
            load_images(),
            create_output_vectors(),
            random_state=0)
        self.cnn()
        self.train_network()


def main():
    ModelTrainer()


if __name__ == "__main__":
    main()
