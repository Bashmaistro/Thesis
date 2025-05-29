import os
import shutil
import numpy as np
import datetime
import cv2
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Function to generate a unique name based on date, number of epochs, and number of training images


def generate_unique_name(base_name, epochs, train_images):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"{base_name}_{current_time}_{epochs}ep_{train_images}img"
    return unique_name


def create_out_folder(date, model_type):
    folder = "results/" + date + "/" + model_type
    if not os.path.isdir(folder):
        os.makedirs(folder)
    models = os.path.join(folder, "models/")
    if not os.path.isdir(models):
        os.makedirs(models)
    plots = os.path.join(folder, "plots/")
    if not os.path.isdir(plots):
        os.makedirs(plots)
    report = folder

    return models, plots, report

def show_features(image, window_name="Image Frames", delay=100):
    """
    Display each frame of a 3D image array frame-by-frame using OpenCV.
    
    :param image: 3D numpy array (Height x Width x Frames)
    :param window_name: Name of the OpenCV window
    :param delay: Delay in milliseconds between frames (default is 100ms)
    """
  # Ensure input image is 3D and has the correct shape
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3D numpy array (Frames x Height x Width).")

    # Loop through each frame
    for i in range(image.shape[0]):
        frame = image[i]

        # Normalize the frame to 8-bit for display
        frame_normalized = cv2.normalize(frame, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Display the frame
        cv2.imshow(window_name, frame_normalized)

        # Wait for the specified delay or until a key is pressed
        key = cv2.waitKey(delay)

        # Exit loop if 'q' is pressed
        if key == ord('q'):
            break

    # Close the OpenCV window
    cv2.destroyAllWindows()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def standardization(npy):
    npy = ((npy-npy.mean())/npy.std())
    return npy


def normalize(npy):
    npy = ((npy-np.amin(npy))/(np.amax(npy)-np.amin(npy)))
    return npy


def create_destroy_dic(path):
    if (not os.path.isdir(path)):
        os.mkdir(path)
    else:
        shutil.rmtree(path)
        os.mkdir(path)

def stack_3(ct):
    ct = np.stack((ct,)*3, axis=-1)
    return ct

def save_numpy(data, path, i=None):
    if i == None:
        if not os.path.isfile(path + '.npy'):
            np.save(path + '.npy', data)
            return
        else:
            i = 1
            save_numpy(data, path, i)
    else:
        name = path + "(" + str(i) + ")" + '.npy'
        if not os.path.isfile(name):
            np.save(name, data)
            return
        else:
            i = i + 1
            save_numpy(data, path, i)
