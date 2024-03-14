import matplotlib.pyplot as plt
import numpy as np

from pertubations import get_augmentations


def plot_distortion_vs_accuracy(title, distortion_levels, accuracies_model1, accuracies_model2, accuracies_model3,
                                model_names,
                                save_path=None):
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 5))

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    plt.plot(distortion_levels, accuracies_model1, color='C0', marker="s", alpha=0.7, label=model_names[0], linewidth=2,
             linestyle="-")

    plt.plot(distortion_levels, accuracies_model2, color='C2', marker="s", alpha=0.7, label=model_names[1], linewidth=2,
             linestyle="-")

    plt.plot(distortion_levels, accuracies_model3, color='C3', marker="s", alpha=0.7, label=model_names[2], linewidth=2,
             linestyle="-")

    plt.title(f'{title} vs. Accuracy', fontsize=18)
    plt.xlabel(title)
    plt.ylabel('mean Average Precision ($mAP_{t=0.5}$)')
    plt.ylim([0, 0.5])
    plt.grid(True)
    plt.legend()
    if save_path:
        plt.savefig(save_path, transparent=True, bbox_inches='tight')
    else:
        plt.show()


def export_performance_plots():
    result_path = 'temp'
    model_names = ['Faster R-CNN ResNet-50-FPN', 'Faster R-CNN ResNet-101-FPN', 'Faster R-CNN ResNet-152-FPN']

    export_contrast(result_path, model_names)
    export_defocus_blur(result_path, model_names)
    export_gaussian_intensity_noise(result_path, model_names)
    export_gaussian_saturation_noise(result_path, model_names)
    export_pepper_noise(result_path, model_names)
    export_rgb_noise(result_path, model_names)

def export_defocus_blur(result_path, model_names):
    defocus_blur = np.load(f"{result_path}/faster_rcnn_resnet50_defocus_blur_x.npy", allow_pickle=True)

    faster_rcnn_resnet50_defocus_blur_y = np.load(f"{result_path}/faster_rcnn_resnet50_defocus_blur_y.npy", allow_pickle=True)
    faster_rcnn_resnet101_defocus_blur_y = np.load(
        f"{result_path}/faster_rcnn_resnet101_defocus_blur_y.npy", allow_pickle=True)
    faster_rcnn_resnet152_defocus_blur_y = np.load(
        f"{result_path}/faster_rcnn_resnet152_defocus_blur_y.npy", allow_pickle=True)

    save_to = f"D:/repository/sensetivity_cnn/example/defocus_blur.jpg"
    plot_distortion_vs_accuracy(
        "Gaus Intensity Noise",
        defocus_blur,
        faster_rcnn_resnet50_defocus_blur_y,
        faster_rcnn_resnet101_defocus_blur_y,
        faster_rcnn_resnet152_defocus_blur_y,
        model_names,
        save_path=save_to,
    )


def export_contrast(result_path, model_names):
    contrast = np.load(f"{result_path}/faster_rcnn_resnet50_contrast_x.npy",
                                       allow_pickle=True)

    faster_rcnn_resnet50_contrast_y = np.load(
        f"{result_path}/faster_rcnn_resnet50_contrast_y.npy",
        allow_pickle=True)
    faster_rcnn_resnet101_contrast_y = np.load(
        f"{result_path}/faster_rcnn_resnet101_contrast_y.npy", allow_pickle=True)
    faster_rcnn_resnet152_contrast_y = np.load(
        f"{result_path}/faster_rcnn_resnet152_contrast_y.npy", allow_pickle=True)

    save_to = f"D:/repository/sensetivity_cnn/example/contrast.jpg"
    plot_distortion_vs_accuracy(
        "Gaus Intensity Noise",
        contrast,
        faster_rcnn_resnet50_contrast_y,
        faster_rcnn_resnet101_contrast_y,
        faster_rcnn_resnet152_contrast_y,
        model_names,
        save_path=save_to,
    )


def export_gaussian_intensity_noise(result_path, model_names):
    gaussian_intensity_noise = np.load(f"{result_path}/faster_rcnn_resnet50_gaussian_intensity_noise_x.npy",
                                       allow_pickle=True)

    faster_rcnn_resnet50_gaussian_intensity_noise_y = np.load(
        f"{result_path}/faster_rcnn_resnet50_gaussian_intensity_noise_y.npy",
        allow_pickle=True)
    faster_rcnn_resnet101_gaussian_intensity_noise_y = np.load(
        f"{result_path}/faster_rcnn_resnet101_gaussian_intensity_noise_y.npy", allow_pickle=True)
    faster_rcnn_resnet152_gaussian_intensity_noise_y = np.load(
        f"{result_path}/faster_rcnn_resnet152_gaussian_intensity_noise_y.npy", allow_pickle=True)

    save_to = f"D:/repository/sensetivity_cnn/example/gaussian_intensity_noise.jpg"
    plot_distortion_vs_accuracy(
        "Gaus Intensity Noise",
        gaussian_intensity_noise,
        faster_rcnn_resnet50_gaussian_intensity_noise_y,
        faster_rcnn_resnet101_gaussian_intensity_noise_y,
        faster_rcnn_resnet152_gaussian_intensity_noise_y,
        model_names,
        save_path=save_to,
    )


def export_gaussian_saturation_noise(result_path, model_names):
    gaussian_saturation_noise = np.load(f"{result_path}/faster_rcnn_resnet50_gaussian_saturation_noise_x.npy",
                                        allow_pickle=True)
    faster_rcnn_resnet50_gaussian_saturation_noise_y = np.load(
        f"{result_path}/faster_rcnn_resnet50_gaussian_saturation_noise_y.npy", allow_pickle=True)
    faster_rcnn_resnet101_gaussian_saturation_noise_y = np.load(
        f"{result_path}/faster_rcnn_resnet101_gaussian_saturation_noise_y.npy", allow_pickle=True)
    faster_rcnn_resnet152_gaussian_saturation_noise_y = np.load(
        f"{result_path}/faster_rcnn_resnet152_gaussian_saturation_noise_y.npy", allow_pickle=True)

    save_to = f"D:/repository/sensetivity_cnn/example/gaussian_saturation_noise.jpg"
    plot_distortion_vs_accuracy(
        "Gausian Saturation Noise",
        gaussian_saturation_noise,
        faster_rcnn_resnet50_gaussian_saturation_noise_y,
        faster_rcnn_resnet101_gaussian_saturation_noise_y,
        faster_rcnn_resnet152_gaussian_saturation_noise_y,
        model_names,
        save_path=save_to,
    )


def export_rgb_noise(result_path, model_names):
    rgb_noise = np.load(f"{result_path}/faster_rcnn_resnet50_rgb noise_x.npy", allow_pickle=True)

    faster_rcnn_resnet50_rgb_noise_y = np.load(f"{result_path}/faster_rcnn_resnet50_rgb noise_y.npy", allow_pickle=True)
    faster_rcnn_resnet101_rgb_noise_y = np.load(f"{result_path}/faster_rcnn_resnet101_rgb noise_y.npy",
                                                allow_pickle=True)
    faster_rcnn_resnet152_rgb_noise_y = np.load(f"{result_path}/faster_rcnn_resnet152_rgb noise_y.npy",
                                                allow_pickle=True)

    save_to = f"D:/repository/sensetivity_cnn/example/rgb_noise.jpg"
    plot_distortion_vs_accuracy(
        "RGB Noise",
        rgb_noise,
        faster_rcnn_resnet50_rgb_noise_y,
        faster_rcnn_resnet101_rgb_noise_y,
        faster_rcnn_resnet152_rgb_noise_y,
        model_names,
        save_path=save_to,
    )


def export_pepper_noise(result_path, model_names):
    pepper_noise = np.load(f"{model_names}/faster_rcnn_resnet50_pepper_noise_x.npy", allow_pickle=True)

    faster_rcnn_resnet50_pepper_noise_y = np.load(f"{result_path}/faster_rcnn_resnet50_pepper_noise_y.npy",
                                                  allow_pickle=True)
    faster_rcnn_resnet101_pepper_noise_y = np.load(f"{result_path}/faster_rcnn_resnet101_pepper_noise_y.npy",
                                                   allow_pickle=True)
    faster_rcnn_resnet152_pepper_noise_y = np.load(f"{result_path}/faster_rcnn_resnet152_pepper_noise_y.npy",
                                                   allow_pickle=True)

    save_to = f"D:/repository/sensetivity_cnn/example/pepper_noise.jpg"
    plot_distortion_vs_accuracy(
        "Pepper Noise",
        pepper_noise,
        faster_rcnn_resnet50_pepper_noise_y,
        faster_rcnn_resnet101_pepper_noise_y,
        faster_rcnn_resnet152_pepper_noise_y,
        model_names,
        save_path=save_to,
    )


if __name__ == "__main__":
    export_performance_plots()
