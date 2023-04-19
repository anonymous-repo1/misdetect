import os
import torch.nn as nn
import argparse
import model as nets
import datasets as datasets
from torchvision import transforms
from utils import *
from datetime import datetime
from time import sleep
import config



def main(args=None):
    if args is None:
        args = config.get_options()
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.train_batch is None:
        args.train_batch = args.batch
    if args.selection_batch is None:
        args.selection_batch = args.batch
    if args.save_path != "" and not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if args.resume != "":
        # Load checkpoint
        # try:
        #     print("=> Loading checkpoint '{}'".format(args.resume))
        #     checkpoint = torch.load(args.resume, map_location=args.device)
        #     assert {"exp", "epoch", "state_dict", "opt_dict", "best_acc1", "rec", "subset", "sel_args"} <= set(
        #         checkpoint.keys())
        #     assert 'indices' in checkpoint["subset"].keys()
        #     start_exp = checkpoint['exp']
        #     start_epoch = checkpoint["epoch"]
        # except AssertionError:
        #     try:
        #         assert {"exp", "subset", "sel_args"} <= set(checkpoint.keys())
        #         assert 'indices' in checkpoint["subset"].keys()
        #         print("=> The checkpoint only contains the subset, training will start from the begining")
        #         start_exp = checkpoint['exp']
        #         start_epoch = 0
        #     except AssertionError:
        #         print("=> Failed to load the checkpoint, an empty one will be created")
        #         checkpoint = {}
        #         start_exp = 0
        #         start_epoch = 0
    # else:
    #     checkpoint = {}
    #     start_exp = 0
    #     start_epoch = 0

    if args.save_path != "":
        checkpoint_name = "{dst}_{net}_epoch{epc}_".format(dst=args.dataset,net=args.model,dat=datetime.now(),epc=args.epochs,)

        print('\n================== Exp %d ==================\n' % exp)
        print("dataset: ", args.dataset, ", model: ", args.model, ", epochs: ", args.epochs, ", mis_ratio: ", args.mis_ratio, ", seed: ", args.seed,
              ", lr: ", args.lr, ", save_path: ", args.save_path, ", mis_distribution: ", args.mis_distribution, ", device: ", args.device,
              ", checkpoint_name: " + checkpoint_name if args.save_path != "" else "", "\n", sep="")


        channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test = datasets.__dict__[args.dataset] \
            (args.data_path)
        args.channel, args.im_size, args.num_classes, args.class_names = channel, im_size, num_classes, class_names
        print(f"channel is [{channel}] im_size (dim) is [{im_size}]")
        print("num_classes is ", num_classes)
        torch.random.manual_seed(args.seed)


        # Augmentation
        if args.dataset == "CIFAR10" or args.dataset == "CIFAR100":
            dst_train.transform = transforms.Compose(
                [transforms.RandomCrop(args.im_size, padding=4, padding_mode="reflect"),
                 transforms.RandomHorizontalFlip(), dst_train.transform])
        elif args.dataset == "KMINIST":
            dst_train.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])

        train_loader = torch.utils.data.DataLoader(dst_subset, batch_size=args.train_batch, shuffle=True,
                                                       num_workers=args.workers, pin_memory=True)
        # test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.train_batch, shuffle=False,
        #                                           num_workers=args.workers, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dst_test, batch_size=args.test_batch, shuffle=False,
                                                      num_workers=args.workers, pin_memory=True)


        # Listing cross-architecture experiment settings if specified.
        models = [args.model]
        if isinstance(args.cross, list):
            for model in args.cross:
                if model != args.model:
                    models.append(model)

        for model in models:

            if len(models) > 1:
                print("| Training on model %s" % model)



            network = nets.__dict__[model](channel, num_classes, im_size).to(args.device)

            print("【See model architecture】")
            print(network)

            if args.device == "cpu":
                print("Using CPU.")
            elif args.gpu is not None:
                torch.cuda.set_device(args.gpu[0])
                network = nets.nets_utils.MyDataParallel(network, device_ids=args.gpu)
            elif torch.cuda.device_count() > 1:
                network = nets.nets_utils.MyDataParallel(network).cuda()

            if "state_dict" in checkpoint.keys():
                # Loading model state_dict
                network.load_state_dict(checkpoint["state_dict"])

            criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)

            # Optimizer
            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(network.parameters(), args.lr, momentum=args.momentum,
                                            weight_decay=args.weight_decay, nesterov=args.nesterov)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(network.parameters(), args.lr, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.__dict__[args.optimizer](network.parameters(), args.lr, momentum=args.momentum,
                                                                 weight_decay=args.weight_decay, nesterov=args.nesterov)

            # Log recorder
            if "rec" in checkpoint.keys():
                rec = checkpoint["rec"]
            else:
                rec = init_recorder()

            best_prec1 = checkpoint["best_acc1"] if "best_acc1" in checkpoint.keys() else 0.0

            # Save the checkpont.
            if args.save_path != "" and args.resume == "":
                save_checkpoint({"exp": exp,
                                 "subset": subset,
                                 "sel_args": selection_args},
                                os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
                                             + "_") + "unknown.ckpt"), 0, 0.)
                coreset_save_path = os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model
                                             + "_") + "unknown.ckpt")


            # tot_epoch = 2
            # if args.dataset == "IMDBLarge":
            #     tot_epoch = 1
            # for epoch in range(start_epoch, tot_epoch):
            # # for epoch in range(start_epoch, args.epochs):
            #     train_test_in_epoch(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec,
            #                         test_loader, args.test_itr_interval, if_weighted=if_weighted)

            for epoch in range(start_epoch, args.epochs):
                # train for one epoch
                train(train_loader, network, criterion, optimizer, scheduler, epoch, args, rec, if_weighted=if_weighted)

                # evaluate on detection accuracy
                if args.test_interval > 0 and (epoch + 1) % args.test_interval == 0:
                    prec1 = test(test_loader, network, criterion, epoch, args, rec)

                    # remember best prec@1 and save checkpoint
                    is_best = prec1 > best_prec1

                    if is_best:
                        best_prec1 = prec1
                        # if args.save_path != "":
                        #     rec = record_ckpt(rec, epoch)
                            # save_checkpoint({"exp": exp,
                            #                  "epoch": epoch + 1,
                            #                  "state_dict": network.state_dict(),
                            #                  "opt_dict": optimizer.state_dict(),
                            #                  "best_acc1": best_prec1,
                            #                  "rec": rec,
                            #                  "subset": subset,
                            #                  "sel_args": selection_args},
                            #                 os.path.join(args.save_path, checkpoint_name + (
                            #                     "" if model == args.model else model + "_") + "unknown.ckpt"),
                            #                 epoch=epoch, prec=best_prec1)



            # Prepare for the next checkpoint
            # if args.save_path != "":
            #     try:
            #         os.rename(
            #             os.path.join(args.save_path, checkpoint_name + ("" if model == args.model else model + "_") +
            #                          "unknown.ckpt"), os.path.join(args.save_path, checkpoint_name +
            #                          ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1))
            #     except:
            #         save_checkpoint({"exp": exp,
            #                          "epoch": args.epochs,
            #                          "state_dict": network.state_dict(),
            #                          "opt_dict": optimizer.state_dict(),
            #                          "best_acc1": best_prec1,
            #                          "rec": rec,
            #                          "subset": subset,
            #                          "sel_args": selection_args},
            #                         os.path.join(args.save_path, checkpoint_name +
            #                                      ("" if model == args.model else model + "_") + "%f.ckpt" % best_prec1),
            #                         epoch=args.epochs - 1,
            #                         prec=best_prec1)
            print("#"*20, "   See rec   ", "#"*20)
            print(rec)
            print('| Best detect accuracy: ', best_prec1, ", on model " + model if len(models) > 1 else "", end="\n\n")
            start_epoch = 0
            checkpoint = {}
            sleep(2)
        train_time = time.time() - st_time
        total_time = selection_time + train_time
        print(f"Total time is  [{total_time}]")
        if args.resume != "":
            coreset_save_path = args.resume
        return best_prec1, total_time, coreset_save_path

if __name__ == '__main__':
    main()
