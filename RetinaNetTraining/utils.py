import argparse
from os.path import join

import numpy as np
from matplotlib import pyplot as plt, patches


def plot_losses(training_losses, training_cls_losses, training_bbox_losses, validation_losses, validation_cls_losses,
                validation_bbox_losses, training_learning_rates, save_path):
    """
    Plot the training losses (combined, cls, and bbox) and validation losses (combined, cls, and bbox) along with the
    learning rates. The figure will be saved at save_path/losses.png. The losses and rates should be in a list
    that grows as the epochs increase.

    :param training_losses: List of training losses.
    :param training_cls_losses:
    :param training_bbox_losses:
    :param validation_losses: List of validation losses.
    :param validation_cls_losses:
    :param validation_bbox_losses:
    :param training_learning_rates: List of optimiser learning rates.
    :param save_path: Directory to save image into.
    """
    epochs = range(1, len(training_losses) + 1)
    _, ax = plt.subplots(nrows=2, ncols=3, layout='constrained', figsize=(16, 9), dpi=200)
    ax[0, 0].set_title('Training Losses (weighted)\n'
                       'with Learning Rate')
    ax[0, 0].plot(epochs, training_losses, marker='*')
    ax[0, 0].set_xlabel('Epoch')
    ax[0, 0].set_ylabel('Loss')
    ax_lr = ax[0, 0].twinx()
    ax_lr.plot(epochs, [i * 100 for i in training_learning_rates], color='red', label='learning rate')
    ax_lr.set_ylabel('Learning Rate x10$^-2$')
    ax_lr.legend(loc='upper right')

    ax[0, 1].set_title('Training Classification Losses')
    ax[0, 1].plot(epochs, training_cls_losses, marker='*')
    ax[0, 1].set_xlabel('Epoch')
    ax[0, 1].set_ylabel('Loss')

    ax[0, 2].set_title('Training Box Regression Losses')
    ax[0, 2].plot(epochs, training_bbox_losses, marker='*')
    ax[0, 2].set_xlabel('Epoch')
    ax[0, 2].set_ylabel('Loss')

    ax[1, 0].set_title('Validation Losses (unweighted)')
    ax[1, 0].plot(epochs, validation_losses, marker='*')
    ax[1, 0].set_xlabel('Epoch')
    ax[1, 0].set_ylabel('Loss')

    ax[1, 1].set_title('Validation Classification Losses')
    ax[1, 1].plot(epochs, validation_cls_losses, marker='*')
    ax[1, 1].set_xlabel('Epoch')
    ax[1, 1].set_ylabel('Loss')

    ax[1, 2].set_title('Validation Box Regression Losses')
    ax[1, 2].plot(epochs, validation_bbox_losses, marker='*')
    ax[1, 2].set_xlabel('Epoch')
    ax[1, 2].set_ylabel('Loss')

    plt.savefig(join(save_path, 'losses.png'))
    plt.close()


def get_arg_parser():
    """
    Set up the parser for the main script.

    :return: parser.parse_args()
    """
    # Path parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-tp',
                        '--train_path',
                        type=str,
                        required=True,
                        help='Path to training folder')
    parser.add_argument('-vp',
                        '--val_path',
                        type=str,
                        required=True,
                        help='Path to validation folder')
    parser.add_argument('-sp',
                        '--save_path',
                        type=str,
                        required=True,
                        help='Path to save folder')
    # Training parameters.
    parser.add_argument('-e',
                        '--epochs',
                        type=int,
                        default=300,
                        help='Training epochs')
    parser.add_argument('-is',
                        '--image_size',
                        type=int,
                        default=600,
                        help='Scaled image size (image_size, image_size)')
    parser.add_argument('-bs',
                        '--batch_size',
                        type=int,
                        default=4,
                        help='Training batch size')
    parser.add_argument('-p',
                        '--patience',
                        type=int,
                        default=100,
                        help='Training patience before early stopping')
    parser.add_argument('-pd',
                        '--patience_delta',
                        type=int,
                        default=0.2,
                        help='Training patience delta')
    parser.add_argument('-lr',
                        '--learning_rate',
                        type=float,
                        default=0.01,
                        help='Optimiser learning rate')
    parser.add_argument('-m',
                        '--momentum',
                        type=float,
                        default=0.9,
                        help='Optimiser momentum')
    parser.add_argument('-wd',
                        '--weight_decay',
                        type=float,
                        default=0.005,
                        help='Optimiser weight decay')
    parser.add_argument('-bw',
                        '--box_weight',
                        type=float,
                        default=7.5,
                        help='Weight applied to box loss')
    parser.add_argument('-cw',
                        '--cls_weight',
                        type=float,
                        default=0.5,
                        help='Weight applied to classification loss')

    return parser.parse_args()


def plot_validation_results(validation_detections, validation_images, counter, save_path):
    """
    Draw input images with detected bounding boxes on them. Only the top scoring box is drawn. This function only
    works for prostate only or prostate and bladder detection.

    :param validation_detections: Detection returned by the model in eval() mode.
    :param validation_images: Images that were given to the model for detection.
    :param counter: Image counter, based on batch_size.
    :param save_path: Save directory.
    """
    batch_number = counter
    for index, output in enumerate(validation_detections):
        top_box_prostate = None
        top_box_bladder = None
        boxes = output['boxes']
        labels = output['labels']

        for i in range(len(labels)):
            if labels[i] == 0 and top_box_prostate is None:
                top_box_prostate = boxes[i].to('cpu')
            elif labels[i] == 1 and top_box_bladder is None:
                top_box_bladder = boxes[i].to('cpu')

            # Break the loop if both top boxes are found.
            if top_box_prostate is not None and top_box_bladder is not None:
                break

        _, ax = plt.subplots()
        ax.axis('off')
        ax.imshow(np.transpose(validation_images[index].to('cpu'), (1, 2, 0)))
        if top_box_prostate is not None:
            prostate_patch = patches.Rectangle((top_box_prostate[0], top_box_prostate[1]),
                                               top_box_prostate[2] - top_box_prostate[0],
                                               top_box_prostate[3] - top_box_prostate[1], linewidth=1,
                                               edgecolor='g', facecolor='none')
            ax.add_patch(prostate_patch)
        if top_box_bladder is not None:
            bladder_patch = patches.Rectangle((top_box_bladder[0], top_box_bladder[1]),
                                              top_box_bladder[2] - top_box_bladder[0],
                                              top_box_bladder[3] - top_box_bladder[1], linewidth=1,
                                              edgecolor='blue', facecolor='none')
            ax.add_patch(bladder_patch)

        plt.savefig(join(save_path, f'val_result_{batch_number}.png'))
        plt.close()
        batch_number += 1
