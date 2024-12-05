"""
Train a RetinaNet model using retinanet_resnet50_fpn_v2.

Transformers, datasets, dataloaders, training, and validation are all taken care of here. Validation is
carried out at each epoch to help prevent over fitting during k-fold cross-validation. When all
the data is used for training, all the same data is used during validation, which is not ideal so that may need
to be fixed in the future, but it seems fine for now.
"""
import os
from datetime import datetime
from os.path import join

import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

import CustomRetinaNet
import utils
from EarlyStopping import EarlyStopping
from ProstateDataset import ProstateDataset

########################################################################################################################
# Set seeds for semi-reproducibility.
########################################################################################################################
script_start = datetime.now()
torch.manual_seed(2023)
torch.cuda.empty_cache()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
########################################################################################################################
# Get parser.
########################################################################################################################
args = utils.get_arg_parser()

########################################################################################################################
# Set up variables from parser.
########################################################################################################################
train_path = args.train_path  # Path to training folder.
val_path = args.val_path  # Path to validation folder.
save_path = args.save_path  # Path to saving folder, where models and epoch loss plots are stored.
if not os.path.isdir(save_path):
    os.makedirs(save_path)  # Make save_path into dir.
image_size = args.image_size  # Image size for resizing, used in training and validation.
batch_size = args.batch_size  # Batch size for loader.
total_epochs = args.epochs  # Training epochs.
warmup_epochs = args.warmup_epochs  # Epochs before early stop checks are done.
patience = args.patience  # Early stopping patience.
patience_delta = args.patience_delta  # Early stopping delta.
learning_rate = args.learning_rate  # Optimiser learning rate.
learning_restart = args.learning_restart  # Learning rate schedular restart frequency.
momentum = args.momentum  # Optimiser momentum.
weight_decay = args.weight_decay  # Optimiser weight decay.
box_weight = args.box_weight  # Weight applied to box loss.
cls_weight = args.box_weight  # Weight applied to classification loss.
oversampling_factor = args.oversampling_factor  # Oversampling factor.

########################################################################################################################
# Transformers used during data loading.
########################################################################################################################
train_transforms = CustomRetinaNet.get_training_transforms(image_size)
val_transforms = CustomRetinaNet.get_validation_transforms(image_size)

########################################################################################################################
# Set up datasets.
########################################################################################################################
train_dataset = ProstateDataset(root=train_path, transforms=train_transforms, oversampling_factor=oversampling_factor)
val_dataset = ProstateDataset(root=val_path, transforms=val_transforms)

########################################################################################################################
# Set up dataloaders.
########################################################################################################################
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=lambda x: tuple(zip(*x)))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                         collate_fn=lambda x: tuple(zip(*x)))

########################################################################################################################
# Set up model, optimiser, and learning rate scheduler (Custom RetinaNet making use of retinanet_resnet50_fpn_v2).
########################################################################################################################
print('Loading model...', end=' ')
custom_model = CustomRetinaNet.CustomRetinaNet()
custom_model.model.to(device)
optimiser = optim.SGD(custom_model.model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
lr_schedular = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimiser, T_0=learning_restart, eta_min=0)
print(f'Model loaded.')

########################################################################################################################
# Save training parameters to file and display to screen.
########################################################################################################################
with open(join(save_path, 'training_parameters.txt'), 'w') as save_file:
    save_file.write(f'Start time: {script_start}\n'
                    f'Training dataset: {train_path}\n'
                    f'Validation dataset: {val_path}\n'
                    f'Save path: {save_path}\n'
                    f'Batch size: {batch_size}\n'
                    f'Epochs: {total_epochs}\n'
                    f'Warmup Epochs: {warmup_epochs}\n'
                    f'Device: {device}\n'
                    f'Patience: {patience}\n'
                    f'Patience delta: {patience_delta}\n'
                    f'Image Size: {image_size}\n'
                    f'Optimiser learning rate: {learning_rate}.\n'
                    f'Optimiser learning rate restart frequency: {learning_restart}\n'
                    f'Optimiser momentum: {momentum}\n'
                    f'Optimiser weight decay: {weight_decay}\n'
                    f'Total training images in dataset (excluding dataset oversampling): {train_dataset.get_image_count()}\n'
                    f'Total training images in dataset (including dataset oversampling): {train_dataset.__len__()}\n'
                    f'Total validation images in dataset: {val_dataset.__len__()}\n'
                    f'Training Transformer count: {len(train_transforms.transforms)}\n'
                    f'Optimiser: {optimiser.__class__.__name__}\n'
                    f'Learning rate schedular: {lr_schedular.__class__.__name__}\n')

print('=====================================================================================================\n'
      f'Start time: {script_start}.\n'
      f'Training using the dataset at: {train_path}.\n'
      f'Validating using the dataset at: {val_path}.\n'
      f'Saving to: {save_path}.\n'
      f'Batch size: {batch_size}.\n'
      f'Epochs: {total_epochs}.\n'
      f'Warmup Epochs: {warmup_epochs}.\n'
      f'Device: {device}\n'
      f'Patience: {patience}.\n'
      f'Patience delta: {patience_delta}.\n'
      f'Image size: {image_size}.\n'
      f'Optimiser learning rate: {learning_rate}.\n'
      f'Optimiser learning rate restart frequency: {learning_restart}.\n'
      f'Optimiser momentum: {momentum}.\n'
      f'Optimiser weight decay: {weight_decay}.\n'
      f'Total training images in dataset (excluding dataset oversampling): {train_dataset.get_image_count()}.\n'
      f'Total training images in dataset (including dataset oversampling): {train_dataset.__len__()}.\n'
      f'Total validation images in dataset: {val_dataset.__len__()}.\n'
      f'Training Transformer count: {len(train_transforms.transforms)}.')


########################################################################################################################
# Training and validation loop.
########################################################################################################################
def main():
    print(f'Starting training on [{device}]:')
    early_stopping = EarlyStopping(patience=patience, delta=patience_delta)
    training_losses = []
    training_cls_losses = []
    training_bbox_losses = []
    validation_losses = []
    validation_cls_losses = []
    validation_bbox_losses = []
    training_learning_rates = []
    final_epoch_reached = 0
    for epoch in range(total_epochs):
        ################################################################################################################
        # Training step within epoch.
        ################################################################################################################
        custom_model.model.train()
        epoch_train_loss = 0
        epoch_train_cls_loss = 0
        epoch_train_bbox_loss = 0
        for images, targets in train_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimiser.zero_grad()
            loss_dict = custom_model.forward(images, targets)
            # Extract each loss.
            cls_loss = loss_dict['classification']
            bbox_loss = loss_dict['bbox_regression']
            # Calculate total loss, can apply weights here.
            losses = cls_loss * cls_weight + bbox_loss * box_weight
            # Check for NaNs or Infs.
            if torch.isnan(losses).any() or torch.isinf(losses).any():
                print("Loss has NaNs or Infs, skipping this batch")
                continue
            # Calculate gradients.
            losses.backward()
            # Apply gradient clipping.
            clip_grad_norm_(custom_model.model.parameters(), 2)
            optimiser.step()
            # Epoch loss per batch.
            epoch_train_loss += losses.item()
            epoch_train_cls_loss += cls_loss.item()
            epoch_train_bbox_loss += bbox_loss.item()
        # Step schedular once per epoch.
        lr_schedular.step()
        # Average epoch loss per image for all images.
        epoch_train_loss /= len(train_loader)
        epoch_train_cls_loss /= len(train_loader)
        epoch_train_bbox_loss /= len(train_loader)
        training_losses.append(epoch_train_loss)
        training_cls_losses.append(epoch_train_cls_loss)
        training_bbox_losses.append(epoch_train_bbox_loss)
        training_learning_rates.append(lr_schedular.get_last_lr()[0])

        ################################################################################################################
        # Validation step within epoch.
        ################################################################################################################
        custom_model.model.eval()
        epoch_validation_loss = 0
        epoch_validation_cls_loss = 0
        epoch_validation_bbox_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = list(image.to(device) for image in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict, _ = custom_model.forward(images, targets)

                cls_loss = loss_dict['classification']
                bbox_loss = loss_dict['bbox_regression']

                losses = cls_loss + bbox_loss

                epoch_validation_loss += losses.item()
                epoch_validation_cls_loss += cls_loss.item()
                epoch_validation_bbox_loss += bbox_loss.item()

            epoch_validation_loss /= len(val_loader)
            epoch_validation_cls_loss /= len(val_loader)
            epoch_validation_bbox_loss /= len(val_loader)

            validation_losses.append(epoch_validation_loss)
            validation_cls_losses.append(epoch_validation_cls_loss)
            validation_bbox_losses.append(epoch_validation_bbox_loss)

        ################################################################################################################
        # Display training and validation losses.
        ################################################################################################################
        time_now = datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
        print(f"\t{time_now}  -  Epoch {epoch + 1}/{total_epochs}, "
              f"Train Loss: {training_losses[-1]:0.4f}, "
              f"Validation Loss: {validation_losses[-1]:0.4f}, "
              f"Learning Rate: {lr_schedular.get_last_lr()[0]:0.6f},", end=' ', flush=True)

        ################################################################################################################
        # Check for early stopping. If patience reached, model is saved and final plots are made.
        ################################################################################################################
        final_epoch_reached = epoch
        if final_epoch_reached + 1 > warmup_epochs:
            early_stopping(epoch_validation_loss, custom_model.model, epoch, optimiser, save_path)

        utils.plot_losses(early_stopping.best_epoch + 1, training_losses, training_cls_losses, training_bbox_losses,
                          validation_losses, validation_cls_losses, validation_bbox_losses,
                          training_learning_rates, save_path)
        if early_stopping.early_stop:
            print('Patience reached, stopping early.')
            break
        else:
            print()

    ####################################################################################################################
    # On training complete, pass through validation images and plot them using best model (must be reloaded).
    ####################################################################################################################
    custom_model.model.load_state_dict(torch.load(join(save_path, 'model_best.pth'))['model_state_dict'])
    custom_model.model.eval()
    with torch.no_grad():
        counter = 0
        for images, targets in val_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            _, detections = custom_model.forward(images, targets)

            utils.plot_validation_results(detections, images, counter, save_path)

            counter += batch_size

    ####################################################################################################################
    # Save extra parameters to file.
    ####################################################################################################################
    script_end = datetime.now()
    run_time = script_end - script_start
    with open(join(save_path, 'training_parameters.txt'), 'a') as save_file:
        save_file.write(f'Final Epoch Reached: {final_epoch_reached}\n'
                        f"End time: {script_end.strftime('%Y-%m-%d  %H:%M:%S')}\n"
                        f'Total run time: {run_time}')

    print('Training completed.\n'
          f"End time: {script_end.strftime('%Y-%m-%d  %H:%M:%S')}.\n"
          f'Total run time: {run_time}.')
    print('=====================================================================================================')


if __name__ == '__main__':
    main()
