import argparse
import os
import copy
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader

from models import SRCNN
from datasets import TrainDataset, EvalDataset
from utils import AverageMeter, calc_psnr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, required=True)
    # Add other arguments if needed
    args = parser.parse_args()

    train_file = "./91-image_x2.h5"
    outputs_dir = "./outputs"
    scale = 3
    lr_init = 1e-3
    batchsize = 16
    numepochs = 200
    numworkers = 8
    seed = 123

    outputs_dir = os.path.join(outputs_dir, 'x{}'.format(scale))

    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(seed)

    model = SRCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD([
        {'params': model.conv1.parameters()},
        {'params': model.conv2.parameters()},
        {'params': model.conv3.parameters(), 'lr': lr_init * 0.1}
    ], lr=lr_init)

    train_dataset = TrainDataset(train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batchsize,
                                  shuffle=True,
                                  num_workers=numworkers,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0

    for epoch in range(numepochs):
        model.train()
        epoch_losses = AverageMeter()

        for data in train_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            preds = model(inputs)

            loss = criterion(preds, labels)

            epoch_losses.update(loss.item(), len(inputs))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), os.path.join(outputs_dir, 'epoch_{}.pth'.format(epoch)))

        model.eval()
        epoch_psnr = AverageMeter()

        for data in eval_dataloader:
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)

            epoch_psnr.update(calc_psnr(preds, labels), len(inputs))

        print('epoch {}'.format(epoch+1) , 'eval psnr: {:.2f}'.format(epoch_psnr.avg))

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

    print('best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
    torch.save(best_weights, os.path.join(outputs_dir, 'best.pth'))
