import argparse
# from pyvirtualdisplay import Display
# display = Display(visible=1, size=(480, 320))
# display.start()
import numpy as np
import torch
# from toy.value_iteration import *
# from toy.network import AttentionNet
# from toy.envs.fourrooms import Fourrooms
# from toy.envs.fourrooms_withcoin import FourroomsCoin
from fourrooms_env.envs.fourrooms_multicoin import FourroomsMultiCoin
from fourrooms_env.value_iteration import gen_dataset_with_value_iteration
from torch import optim
import os
import cv2
import pickle
# from toy.util import *

# what to do: create an envs(like fourroom, compute the value of its states,
# and train a neural network with mask to see if it can learn desired mask)


# def generate_image(add_noise=False):
#     obs, values = dataset.X, dataset.y
#     predict_values = []
#     feature_maps = []
#     for i, data in enumerate(zip(obs, values)):
#         obs, value = data
#         if add_noise:
#             obs = obs + torch.rand(*(obs.shape)).to(device) * 0.1 - 0.05
#         mask, value_predict, feature_map = value_network(obs[np.newaxis, ...])
#
#         mask = mask.detach().cpu().numpy()[0, 0]
#         mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask) + 1e-12)
#
#         feature_map = feature_map.detach().cpu().numpy()[0, 0]
#         feature_maps.append(feature_map)
#         # if i == 0:
#         #     print(feature_map)
#         predict_values.append(value_predict.detach().cpu().numpy().squeeze())
#         feature_map = (feature_map - np.min(feature_map)) / (np.max(feature_map) - np.min(feature_map) + 1e-12)
#
#         obs = obs.cpu().numpy().transpose((1, 2, 0))
#         if add_noise:
#
#             suffix = "noise"
#         else:
#             suffix = ""
#         obs = obs * 255
#         mask = cv2.resize(mask, (obs.shape[0], obs.shape[1]))
#         mask = np.repeat(mask[..., np.newaxis], 3, axis=2)
#
#         feature_map = cv2.resize(feature_map, (obs.shape[0], obs.shape[1]))
#         feature_map = np.repeat(feature_map[..., np.newaxis], 3, axis=2)
#         # print(self.obs_shape)
#
#         masked_image = obs * mask
#         cv2.imwrite(os.path.join(target_path, "./mask/", "{}_{}.png".format(suffix, i)), mask * 255)
#         cv2.imwrite(os.path.join(target_path, "./image/", "{}_{}.png".format(suffix, i)), obs)
#         cv2.imwrite(os.path.join(target_path, "./masked_image/", "{}_{}.png".format(suffix, i)), masked_image)
#         cv2.imwrite(os.path.join(target_path, "./feature_map/", "{}_{}.png".format(suffix, i)), feature_map * 255)
#     values = values.cpu().numpy()
#     print("mse", np.mean((values - predict_values) ** 2))


def gen_dataset(env, device):
    try:
        num_state = env.observation_space.shape[0]
    except IndexError:
        num_state = env.observation_space.n

    # get observations
    obs = []
    for s in range(num_state):
        env.reset(s)
        obs.append(env.render())
    obs = np.array(obs)

    return

def main(args):
    device = torch.device(args.device)

    # envs = Fourrooms()
    env = FourroomsMultiCoin()

    dataset, transition = gen_dataset_with_value_iteration(env, device)

    # value_network = AttentionNet(input_size=feature_map_size, device=device).to(device)
    # value_network = AttentionNet(input_size=144*2).to(device)

    # optimizer = optim.Adam(value_network.parameters(), lr=lr, weight_decay=1e-5)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True)

    # # train network
    # for epoch in range(1, epochs + 1):
    #     value_network.train()
    #     train_loss = 0
    #     total_correct = 0
    #     for batch_idx, (obs, value_gt) in enumerate(data_loader):
    #         optimizer.zero_grad()
    #         mask, value_predict, _ = value_network(obs)
    #         encoder_loss, decoder_loss = value_network.loss_func(mask, value_predict, value_gt)
    #         obs_tar, obs_pos, obs_neg = sample_contrast(transition, batch_size, dataset)
    #         contrast_loss = value_network.contrast_loss_func(obs_tar, obs_pos, obs_neg) + value_network.contrast_loss_func(obs_pos, obs_tar, obs_neg)
    #
    #         loss = 1e-2*encoder_loss + decoder_loss + contrast_loss
    #         loss.sum().backward()
    #         # total_correct += correct.sum().item()
    #         train_loss += loss.sum().item()
    #         optimizer.step()
    #         if epoch % log_interval == 0:
    #             print('Encoder Loss:{} Decoder Loss:{}'.format(encoder_loss,decoder_loss))
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #                 epoch, batch_idx * len(obs), len(data_loader.dataset),
    #                        100. * batch_idx / len(data_loader),
    #                        loss.sum().item() / len(obs), end='\r'))
    #
    #     # if epoch % 1 == 0:
    #     #         torch.save(model.state_dict(), "./model/FC_{}_epoch{}_predict{}.model".format(b, epoch, args.predict))
    #
    # # print mask
    # target_path = "./attention/"
    # # if not os.path.exists(target_path):
    # os.makedirs(os.path.join(target_path, "./mask/"), exist_ok=True)
    # os.makedirs(os.path.join(target_path, "./image/"), exist_ok=True)
    # os.makedirs(os.path.join(target_path, "./feature_map/"), exist_ok=True)
    # os.makedirs(os.path.join(target_path, "./masked_image/"), exist_ok=True)

    # predict_values = []
    # feature_maps = []

    # generate_image()
    # generate_image(True)
    # print(value_network.value_fc_4.weight.shape)
    # print(np.mean(abs(value_network.value_fc_4.weight.cpu().detach().numpy())))
    # print(value_network.value_fc_4.weight[0,:feature_map_size**2].reshape(feature_map_size,feature_map_size))
    # # print(value_network.value_fc_4.weight[:,feature_map_size**2:].reshape(feature_map_size,feature_map_size))
    # print(np.array(predict_values))
    # feature_maps = np.array(feature_maps)
    # print(feature_maps.shape)
    # pickle.dump(feature_maps, open("feature_map.pkl", "wb"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='Pytorch device')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    main(args)
