import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import cv2
from sklearn.utils import shuffle
from tensorflow.python.framework import ops

ops.reset_default_graph()


# Identifies which class is responsible for image
# Currently we have 3 image types
# Categorial Crossentropy - 3 class identifies
def create_output_vectors():
    output_vectors = []
    for i in range(0, 1000):
        output_vectors.append([1, 0, 0, 0, 0, 0, 0, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 1, 0, 0, 0, 0, 0, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 0, 1, 0, 0, 0, 0, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 0, 0, 1, 0, 0, 0, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 0, 0, 0, 1, 0, 0, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 0, 0, 0, 0, 1, 0, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 0, 0, 0, 0, 0, 1, 0])
    for i in range(0, 1000):
        output_vectors.append([0, 0, 0, 0, 0, 0, 0, 1])
    return output_vectors


# Load dataset images
def load_images():
    loaded_images = []
    # Load images for swing
    for i in range(0, 1000):
        image = cv2.imread('Dataset/SwingImages/swing_' + str(i) + '.png')
        # Convert into grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Reshape image - all images must have the same size
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/PalmImages/palm_' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/FistImages/fist_' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/NoneImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(256, 256, 1))
    for i in range(1, 1001):
        image = cv2.imread('Dataset/OkImages/image (' + str(i) + ').png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/PeaceImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/StraightImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/ThumbsImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(256, 256, 1))
    return loaded_images


# Load test images - Cross validation
# Estimate the skill of a machine learning model on unseen data
def load_test_images():
    test_images = []
    for i in range(0, 100):
        image = cv2.imread('Dataset/SwingTest/swing_' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/PalmTest/palm_' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/FistTest/fist_' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))  # 1 - deklaruje ilość barw
    for i in range(0, 100):
        image = cv2.imread('Dataset/NoneImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/OkTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/PeaceTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/StraightTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(256, 256, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/ThumbsTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(256, 256, 1))
    return test_images


def init_test_labels():
    test_labels = []
    for i in range(0, 100):
        test_labels.append([1, 0, 0, 0, 0, 0, 0, 0])
    for i in range(0, 100):
        test_labels.append([0, 1, 0, 0, 0, 0, 0, 0])
    for i in range(0, 100):
        test_labels.append([0, 0, 1, 0, 0, 0, 0, 0])
    for i in range(0, 100):
        test_labels.append([0, 0, 0, 1, 0, 0, 0, 0])
    for i in range(0, 100):
        test_labels.append([0, 1, 0, 0, 1, 0, 0, 0])
    for i in range(0, 100):
        test_labels.append([0, 0, 0, 0, 0, 1, 0, 0])
    for i in range(0, 100):
        test_labels.append([0, 0, 0, 0, 0, 0, 1, 0])
    for i in range(0, 100):
        test_labels.append([0, 0, 0, 0, 0, 0, 0, 1])
    return test_labels


class ModelTrainer:

    def train_network(self):
        # Train network,
        self.model.fit(self.loadedImages, self.outputVectors, n_epoch=1,
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
        # self.conv_net = conv_2d(self.conv_net, 128, 2, activation='relu')
        # self.conv_net = max_pool_2d(self.conv_net, 2)
        # # 256 -> 256
        # self.conv_net = conv_2d(self.conv_net, 256, 2, activation='relu')
        # self.conv_net = max_pool_2d(self.conv_net, 2)
        # # 256 -> 128
        # self.conv_net = conv_2d(self.conv_net, 256, 2, activation='relu')
        # self.conv_net = max_pool_2d(self.conv_net, 2)
        # # 128 -> 64
        # self.conv_net = conv_2d(self.conv_net, 128, 2, activation='relu')
        # self.conv_net = max_pool_2d(self.conv_net, 2)
        # 64
        self.conv_net = conv_2d(self.conv_net, 64, 2, activation='relu')
        self.conv_net = max_pool_2d(self.conv_net, 2)

        self.conv_net = fully_connected(self.conv_net, 1000, activation='relu')
        self.conv_net = dropout(self.conv_net, 0.75)
        # 64 -> 3
        # 3 neurons on the output
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
