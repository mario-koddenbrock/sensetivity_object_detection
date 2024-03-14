import os

import cv2  # OpenCV for image processing
import numpy as np

import example


def add_rgb_noise(image, intensity=50):
    """
    Add RGB noise to the given image.

    Args:
        image (numpy.ndarray): Input image.
        intensity (int): Intensity of the noise.

    Returns:
        numpy.ndarray: Image with added RGB noise.
    """

    if intensity > 0:
        noise = np.random.randint(-intensity, intensity, image.shape)
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
    else:
        noisy_image = image

    return noisy_image


def add_gaussian_saturation_noise(image, std=25):
    """
    Add Gaussian saturation noise to the given image.

    Args:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Image with added Gaussian saturation noise.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    noise = np.random.normal(0, std, hsv_image[:, :, 1].shape)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] + noise, 0, 255).astype(np.uint8)
    noisy_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return noisy_image


def add_gaussian_intensity_noise(image, std=25):
    """
    Add Gaussian intensity noise to the given image.

    Args:
        image (numpy.ndarray): Input image.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.

    Returns:
        numpy.ndarray: Image with added Gaussian intensity noise.
    """
    # Convert image to YUV color space (or any other suitable color space)
    yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

    # Extract the intensity (Y) channel
    intensity_channel = yuv_image[:,:,0]

    # Generate Gaussian noise with the same shape as the intensity channel
    gaussian_noise = np.random.normal(0, std, size=intensity_channel.shape)

    # Add Gaussian noise to the intensity channel
    noisy_intensity = np.clip(intensity_channel + gaussian_noise, 0, 255).astype(np.uint8)

    # Replace the intensity channel with the noisy intensity
    yuv_image[:,:,0] = noisy_intensity

    # Convert the image back to BGR color space
    noisy_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    return noisy_image


def add_pepper_noise(image, probability=0.01):
    """
    Add pepper noise to the given image.

    Args:
        image (numpy.ndarray): Input image.
        probability (float): Probability of setting a pixel to zero.

    Returns:
        numpy.ndarray: Image with added pepper noise.
    """
    noisy_image = np.copy(image)
    mask = np.random.rand(*image.shape[:2])
    noisy_image[mask < probability] = 0
    return noisy_image


def add_global_color_shift(image, intensity=50):
    """
    Add global color shift to the given image.

    Args:
        image (numpy.ndarray): Input image.
        intensity (int): Intensity of the color shift.

    Returns:
        numpy.ndarray: Image with added global color shift.
    """
    shift = np.random.randint(-intensity, intensity, 3)
    noisy_image = np.clip(image + shift, 0, 255).astype(np.uint8)
    return noisy_image


def add_defocus_blur(image, kernel_size=15):
    """
    Add defocus blur to the given image.

    Args:
        image (numpy.ndarray): Input image.
        kernel_size (int): Size of the defocus blur kernel.

    Returns:
        numpy.ndarray: Image with added defocus blur.
    """

    if kernel_size:
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        defocus_blur_image = cv2.filter2D(image, -1, kernel)
    else:
        defocus_blur_image = image

    return defocus_blur_image


def add_color_aberration(image, shift=5):
    """
    Add color aberration to the given image.

    Args:
        image (numpy.ndarray): Input image.
        shift (float): Intensity of the color aberration.

    Returns:
        numpy.ndarray: Image with added color aberration.
    """
    r, g, b = cv2.split(image)
    shifted_r = np.roll(r, -shift, axis=0)
    shifted_g = np.roll(g, shift, axis=1)
    shifted_b = np.roll(b, shift, axis=0)
    color_aberration_image = cv2.merge((shifted_r, shifted_g, shifted_b))
    return color_aberration_image


def adjust_contrast(image, intensity=1.5):
    """
    Adjust the contrast of the given image.

    Args:
        image (numpy.ndarray): Input image.
        intensity (float): Intensity of the contrast adjustment.

    Returns:
        numpy.ndarray: Image with adjusted contrast.
    """
    adjusted_image = cv2.convertScaleAbs(image, alpha=intensity, beta=0)
    return adjusted_image


def idetety(image, dummy):
    return image


def apply_all_augmentations(image_path, output_folder):
    image = cv2.imread(image_path)

    augmentations = get_augmentations()

    for augmentation_name, augmentation_func, aug_settings in augmentations:
        for param in aug_settings:
            augmented_image = augmentation_func(image, param)
            output_path = os.path.join(output_folder, f"{augmentation_name}_{param:.2f}.jpg")
            cv2.imwrite(output_path, augmented_image)
            example.export_prediction_visualization(output_path)


def get_augmentations():
    num_distortions = 10
    augmentations = [
        # ("color_aberration", add_color_aberration, np.linspace(1, 5, 5, dtype=int)),
        ("contrast", adjust_contrast, np.linspace(0.5, 2, num_distortions)),
        ("defocus_blur", add_defocus_blur, np.linspace(0, 5, 6, dtype=int)),
        ("gaussian_intensity_noise", add_gaussian_intensity_noise, np.linspace(0, 50, num_distortions, dtype=int)),
        ("gaussian_saturation_noise", add_gaussian_saturation_noise, np.linspace(0, 130, num_distortions, dtype=int)),
        # ("global_color_shift", add_global_color_shift, np.linspace(0, 100, num_distortions, dtype=int)),
        ("pepper_noise", add_pepper_noise, np.linspace(0, 0.2, num_distortions)),
        ("rgb_noise", add_rgb_noise, np.linspace(0, 120, num_distortions, dtype=int)),
    ]

    # augmentations = [
    #     ("rgb noise", add_rgb_noise, np.linspace(0, 120, num_distortions, dtype=int)),
    #     ("gaussian_saturation_noise", add_gaussian_saturation_noise, np.linspace(0, 130, num_distortions, dtype=int)),
    #     ("gaussian_intensity_noise", add_gaussian_intensity_noise, np.linspace(0, 50, num_distortions, dtype=int)),
    #     ("pepper_noise", add_pepper_noise, np.linspace(0, 0.2, num_distortions)),
    # ]

    return augmentations


if __name__ == "__main__":
    image_path = "D:/repository/sensetivity_cnn/example/example.jpg"
    output_folder = "D:/repository/sensetivity_cnn/example/augmentations"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    apply_all_augmentations(image_path, output_folder)
