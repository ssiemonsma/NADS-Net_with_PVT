import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from dataset_generator import Dataset_Generator_MS_COCO, CanonicalConfig, COCOSourceConfig
from model import NADS_Net
from tqdm import tqdm
import numpy as np

filename = 'weights_pretraining_with_tiny_PVT.pth'
pretrained_filename = 'weights_pretraining1.pth'

# Training Parameters
start_from_pretrained = False
num_training_epochs = 100
batch_size = 10
starting_lr = 1e-5  # For Adam optimizer
# starting_lr = 2e-6  # For SGD
# lr_schedule_type = 'fixed'
lr_schedule_type = 'metric-based'
lr_gamma = 0.7  # (for both fixed and metric-based scheduler) This is the factor the learning rate decreases by after the metric doesn't improve for some time
patience = 2   # (for metric-based scheduler only) The number of epochs that must pass without metric improvement for the learning rate to decrease
step_size = 20   # (for fixed scheduler only) After this many epochs without improvement, the learning rate is decreased
weight_decay = 5e-7
num_dataloader_threads = 4
include_seatbelt_branch = False
include_background_output = False
using_Aisin_output_format = True

# Instantiate the Network
device = torch.device("cuda:7")
net = NADS_Net(include_seatbelt_branch, using_Aisin_output_format, include_background_output).to(device)
print('Network contains', sum([p.numel() for p in net.parameters()]), 'parameters.')

# Create the Data Loaders
train_dataset = Dataset_Generator_MS_COCO(CanonicalConfig(), COCOSourceConfig("/localscratch/Users/ssiemons/NADS-Net/COCO_Dataset/coco_train_dataset.h5"), using_Aisin_output_format, include_background_output, augment=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_dataloader_threads, pin_memory=True, shuffle=True, drop_last=False)
num_training_samples = len(train_dataset)
print('Number of training samples = %i' % num_training_samples)

valid_dataset = Dataset_Generator_MS_COCO(CanonicalConfig(), COCOSourceConfig("/localscratch/Users/ssiemons/NADS-Net/COCO_Dataset/coco_val_dataset.h5"), using_Aisin_output_format, include_background_output, augment=False)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_dataloader_threads, pin_memory=True, shuffle=False, drop_last=False)
num_validation_samples = len(valid_dataset)
print('Number of validation samples = %i' % num_validation_samples)

# Set up the Optimizer, Loss Function, Learning Rate Scheduler, and Logger
# optimizer = optim.AdamW(net.parameters(), lr=starting_lr, weight_decay=weight_decay)
optimizer = optim.SGD(net.parameters(), lr=starting_lr, momentum=0.9, weight_decay=weight_decay, nesterov=False)

if lr_schedule_type == 'fixed':
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=lr_gamma)
elif lr_schedule_type == 'metric-based':
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_gamma, patience=patience, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-10)

writer = SummaryWriter()

def MSE_criterion(input, target, batch_size):
    return nn.MSELoss(reduction='sum')(input, target)/batch_size

if start_from_pretrained:
    net.load_state_dict(torch.load(pretrained_filename), strict=False)

best_loss = np.float('inf')

# Training Loop
for epoch in range(num_training_epochs):
    # Training
    running_MSE_loss = 0
    running_keypoint_heatmap_MSE_loss = 0
    running_PAF_MSE_loss = 0

    progress_bar = tqdm(train_loader)
    for i, data in enumerate(progress_bar):
        input_images, keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].float().to(device), data[4].float().to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + Backward + Optimize
        keypoint_heatmap, PAFs = net(input_images, keypoint_heatmap_masks, PAF_masks)

        if include_background_output and using_Aisin_output_format:
            # Background layer is not used in algorithm, so we're not going to track its loss
            keypoint_heatmap_MSE_loss = MSE_criterion(keypoint_heatmap[:,:9,:,:], keypoint_heatmap_labels[:,:9,:,:], len(input_images))
        else:
            keypoint_heatmap_MSE_loss = MSE_criterion(keypoint_heatmap, keypoint_heatmap_labels, len(input_images))

        PAF_MSE_loss = MSE_criterion(PAFs, PAF_labels, len(input_images))

        MSE_loss = keypoint_heatmap_MSE_loss + PAF_MSE_loss

        MSE_loss.backward()
        optimizer.step()

        running_MSE_loss += MSE_loss.item()*len(input_images)
        running_keypoint_heatmap_MSE_loss += keypoint_heatmap_MSE_loss.item()*len(input_images)
        running_PAF_MSE_loss += PAF_MSE_loss.item()*len(input_images)

        batch_num = i + epoch*len(train_loader)
        writer.add_scalar("Loss/training/batches/MSE", MSE_loss, batch_num)
        writer.add_scalar("Loss/training/batches/keypoint_heatmap_MSE", keypoint_heatmap_MSE_loss, batch_num)
        writer.add_scalar("Loss/training/batches/PAF_MSE", PAF_MSE_loss, batch_num)

        progress_bar.set_description('Epoch %i Training' % epoch)
        progress_bar.set_postfix_str('Train Batch MSE: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f' % (MSE_loss, keypoint_heatmap_MSE_loss, PAF_MSE_loss))

    epoch_MSE_loss = running_MSE_loss/num_training_samples
    epoch_keypoint_heatmap_MSE_loss = running_keypoint_heatmap_MSE_loss/num_training_samples
    epoch_PAF_MSE_loss = running_PAF_MSE_loss/num_training_samples

    writer.add_scalar("Loss/training/epochs/MSE", epoch_MSE_loss, epoch)
    writer.add_scalar("Loss/training/epochs/keypoint_heatmap_MSE", epoch_keypoint_heatmap_MSE_loss, epoch)
    writer.add_scalar("Loss/training/epochs/PAF_MSE", epoch_PAF_MSE_loss, epoch)

    print('Epoch %i: Training MSE: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f' % (epoch, epoch_MSE_loss, epoch_keypoint_heatmap_MSE_loss, epoch_PAF_MSE_loss))

    # Validation
    with torch.no_grad():
        running_MSE_loss = 0
        running_keypoint_heatmap_MSE_loss = 0
        running_PAF_MSE_loss = 0

        progress_bar = tqdm(valid_loader)
        for i, data in enumerate(progress_bar):
            input_images, keypoint_heatmap_masks, PAF_masks, keypoint_heatmap_labels, PAF_labels = data[0].to(device), data[1].to(device), data[2].to(device), data[3].to(device), data[4].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            keypoint_heatmap, PAFs = net(input_images, keypoint_heatmap_masks, PAF_masks)

            if include_background_output and using_Aisin_output_format:
                # Background layer is not used in algorithm, so we're not going to track its loss
                keypoint_heatmap_MSE_loss = MSE_criterion(keypoint_heatmap[:, :9, :, :], keypoint_heatmap_labels[:, :9, :, :], len(input_images))
            else:
                keypoint_heatmap_MSE_loss = MSE_criterion(keypoint_heatmap, keypoint_heatmap_labels, len(input_images))

            PAF_MSE_loss = MSE_criterion(PAFs, PAF_labels, len(input_images))

            MSE_loss = keypoint_heatmap_MSE_loss + PAF_MSE_loss

            running_MSE_loss += MSE_loss.item()*len(input_images)
            running_keypoint_heatmap_MSE_loss += keypoint_heatmap_MSE_loss.item()*len(input_images)
            running_PAF_MSE_loss += PAF_MSE_loss.item()*len(input_images)

            progress_bar.set_description('Epoch %i Validation' % epoch)
            progress_bar.set_postfix_str('Valid Batch MSE: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f' % (MSE_loss, keypoint_heatmap_MSE_loss, PAF_MSE_loss))

        epoch_MSE_loss = running_MSE_loss/num_validation_samples
        epoch_keypoint_heatmap_MSE_loss = running_keypoint_heatmap_MSE_loss/num_validation_samples
        epoch_PAF_MSE_loss = running_PAF_MSE_loss/num_validation_samples

        writer.add_scalar("Loss/validation/epochs/MSE", epoch_MSE_loss, epoch)
        writer.add_scalar("Loss/validation/epochs/keypoint_heatmap_MSE", epoch_keypoint_heatmap_MSE_loss, epoch)
        writer.add_scalar("Loss/validation/epochs/PAF_MSE", epoch_PAF_MSE_loss, epoch)

        print('Epoch %i: Validation MSE: %.3f, Keypoint heatmap MSE: %.3f, PAF MSE: %.3f' % (epoch, epoch_MSE_loss, epoch_keypoint_heatmap_MSE_loss, epoch_PAF_MSE_loss))

    if epoch_MSE_loss < best_loss:
        print('Best validation loss achieved:', epoch_MSE_loss)
        torch.save(net.state_dict(), filename)
        best_loss = epoch_MSE_loss

    if lr_schedule_type == 'fixed':
        scheduler.step()
    elif lr_schedule_type == 'metric-based':
        scheduler.step(epoch_MSE_loss)
