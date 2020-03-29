import os
import time

import numpy as np
import torch
from torch import optim
from torchvision.utils import save_image

from src.utils.helpers import to_img
from src.utils.measures import tucker_measure, spearman_metric_ilp

REAL_LABEL = 1
FAKE_LABEL = 0


class Trainer:
    def __init__(self, model, loss_class, dataloaders, cuda):
        self.model = model
        self.loss_class = loss_class
        self.train_dataloader, self.test_dataloader = dataloaders
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

    @staticmethod
    def save_images_from_epoch(args, train_imgs, test_imgs, epoch):
        if (epoch + 1) % args.save_every == 0:
            if not args.save_raw:
                # specific for this dataset (selects only the first image)
                save_train = to_img(train_imgs[0:321 * 481].T.reshape(2, 1, 321, 481).cpu().data, args.normalize_img)
                save_test = to_img(test_imgs[0:321 * 481].T.reshape(2, 1, 321, 481).cpu().data, args.normalize_img)
                save_image(save_train,
                           os.path.join(os.path.join(args.folder, 'images'), 'train_image_{}.png'.format(epoch)))
                save_image(save_test,
                           os.path.join(os.path.join(args.folder, 'images'), 'test_image_{}.png'.format(epoch)))
            else:
                print("Unknown or incompatible input data <{}>. Saving raw outputs".format(args.dataset))
                save_train = train_imgs.cpu().data
                save_test = test_imgs.cpu().data
                path_train = os.path.join(os.path.join(args.folder, 'images'), 'train_{}.npy'.format(epoch))
                path_test = os.path.join(os.path.join(args.folder, 'images'), 'test_{}.npy'.format(epoch))
                np.save(path_train, save_train)
                np.save(path_test, save_test)

    def report(self, epoch, epoch_time, loss_dict):
        report_string = '====> Epoch: {} [Time: {:.2f}s] '.format(epoch, epoch_time)
        for key, value in loss_dict.items():
            report_string += '[{}:{:.4f}]. '.format(key, value)
        print(report_string)


class IndependenceAETrainer(Trainer):
    def __init__(self, model, loss_class, dataloaders, cuda):
        super().__init__(model, loss_class, dataloaders, cuda)

    def train(self, args):
        num_epochs = args.num_epochs
        lr = args.lr
        optimizer = optim.Adam(self.model.parameters(), lr)

        print("Beginning training...")
        for epoch in range(num_epochs):
            self.model.train()
            train_loss, train_sprm, train_tucker = 0, 0, 0

            start = time.time()

            images_to_save = []

            for batch_idx, (data, orig) in enumerate(self.train_dataloader):
                if args.cuda:
                    data = data.cuda()
                    orig = orig.cuda()
                optimizer.zero_grad()
                loss, recon, encoded = self.calculate_loss(data)

                images_to_save.extend(encoded)

                loss.backward()
                train_loss += loss.data
                train_tucker += self.calculate_tucker(encoded, orig)
                train_sprm += self.calculate_spearman(encoded, orig)
                optimizer.step()
            end = time.time()

            recon = torch.stack(images_to_save, dim=0)

            self.model.eval()
            test_loss, test_sprm, test_tucker = 0, 0, 0

            images_to_save = []

            for batch_idx, (data, t_orig) in enumerate(self.test_dataloader):
                if args.cuda:
                    data = data.cuda()
                loss, t_recon, t_encoded = self.calculate_loss(data)

                images_to_save.extend(t_encoded)

                if epoch == num_epochs - 1 and args.save_raw:
                    save_test = t_encoded.cpu().data
                    path_test = os.path.join(os.path.join(args.folder, 'images'),
                                             'icatest_{}_{}.npy'.format(batch_idx, epoch))
                    np.save(path_test, save_test)
                test_loss += loss.data
                test_tucker += self.calculate_tucker(t_encoded, t_orig)
                test_sprm += self.calculate_spearman(t_encoded, t_orig)

            t_recon = torch.stack(images_to_save, dim=0)

            epoch_time = end - start
            loss_dict = self.get_loss_dict(
                train_loss,
                test_loss,
                train_tucker,
                test_tucker,
                train_sprm,
                test_sprm
            )
            self.report(epoch, epoch_time, loss_dict)
            self.save_images_from_epoch(args, recon, t_recon, epoch)

        state = self.get_state_dict(epoch, optimizer, loss_dict)
        torch.save(state, os.path.join(args.folder, "model.th".format(epoch)))
        print("Training complete")

    def calculate_tucker(self, x, y):
        return tucker_measure(x.detach().numpy(), y.detach().numpy())

    def calculate_spearman(self, x, y):
        return spearman_metric_ilp(x.detach().numpy(), y.detach().numpy())

    def get_state_dict(self, epoch, optimizer, loss_dict):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        state.update(loss_dict)
        return state

    def calculate_loss(self, img):
        recon, encoded = self.model(img)
        loss = self.loss_class.loss(recon, img)
        return loss, recon, encoded

    def get_loss_dict(self, train_loss, test_loss, train_tucker, test_tucker, train_sprm, test_sprm):
        div_train, div_test = len(self.train_dataloader), len(self.test_dataloader)

        return {
            'rec_loss': train_loss / div_train,
            'rec_loss_test': test_loss / div_test,
            'train_tucker': train_tucker / div_train,
            'test_tucker': test_tucker / div_test,
            'train_sprm': 1 - train_sprm / div_train,
            'test_sprm': 1 - test_sprm / div_test,
        }

    def calculate_loss(self, img):
        recon, encoded = self.model(img)
        loss = self.loss_class.loss(recon, img, z=encoded)
        return loss, recon, encoded
