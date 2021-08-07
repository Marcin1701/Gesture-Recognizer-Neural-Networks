import cv2


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
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(1, 1001):
        image = cv2.imread('Dataset/OkImages/image (' + str(i) + ').png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/PeaceImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/StraightImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 1000):
        image = cv2.imread('Dataset/ThumbsImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        loaded_images.append(gray_image.reshape(89, 100, 1))
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
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/NoneImages/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/OkTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/PeaceTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/StraightTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
    for i in range(0, 100):
        image = cv2.imread('Dataset/ThumbsTest/' + str(i) + '.png')
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        test_images.append(gray_image.reshape(89, 100, 1))
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
