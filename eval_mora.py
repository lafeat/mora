import os
import argparse
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import sys
from mora import MORA
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/cifar10')
    parser.add_argument('--norm', type=str, default='Linf')
    parser.add_argument('--epsilon', type=float, default=8. / 255.)
    parser.add_argument('--model_file', type=str, default='./frize_sat_0_88.81999969482422.pth.tar')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./log_file.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--float_dis', type=float, default=0.2)
    parser.add_argument('--n_iter', type=int, default=100)
    parser.add_argument('--arch', default='ResNet', type=str, choices=['ResNet'], help='model architecture')
    parser.add_argument('--depth', default=20, type=int, help='depth of the model')
    parser.add_argument('--scale', action='store_true')
    parser.add_argument('--decay_step', type=str, default='linear')
    parser.add_argument('--gpu', type=str, default=0)
    parser.add_argument('--model_name', type=str, default='dverge')
    parser.add_argument('--lr_start', type=float, default=1.0)
    parser.add_argument('--ensemble_pattern', type=str, default='softmax')

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False

    if args.ensemble_pattern == 'softmax':
        from models.ensemble_softmax import Ensemble
    elif args.ensemble_pattern == 'logits':
        from models.ensemble_logits import Ensemble
    elif args.ensemble_pattern == 'voting':
        from models.ensemble_voting import Ensemble

    models = utils.get_models(args, train=False, as_ensemble=False, model_file=args.model_file, leaky_relu=leaky_relu)
    model = Ensemble(models)

    model.cuda()
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)
    item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    test_loader = data.DataLoader(item, batch_size=1000, shuffle=False, num_workers=0)

    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # load attack
    adversary = MORA(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path, version=args.version,
                           scale=args.scale, decay_step=args.decay_step, n_iter=args.n_iter,
                           float_dis=args.float_dis, model_name=args.model_name, 
                           ensemble_pattern=args.ensemble_pattern)

    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)

    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['mora-ce']
        adversary.mora.n_restarts = 2
        adversary.fab.n_restarts = 2

    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                                                             bs=args.batch_size)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'mora', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                                                                        y_test[:args.n_ex], bs=args.batch_size)

            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'mora', args.version, args.n_ex, args.epsilon))
