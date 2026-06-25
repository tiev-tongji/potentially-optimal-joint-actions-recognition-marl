import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.pow_mixer import PowMixer
from modules.mixers.qmix_sa import QMixerSA
from modules.mixers.vdn import VDNMixer
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.qmix_central_no_hyper import QMixerCentralFF
from controllers import REGISTRY as mac_REGISTRY
import torch.nn.functional as F
import torch as th
from torch.optim import Adam
import numpy as np
from utils.rl_utils import build_td_lambda_targets
from utils.th_utils import get_parameters_num
from envs.matrix_game import print_matrix_status


class PowqmixLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "pow_mixer":
                self.mixer = PowMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.qmixer_sa = QMixerSA(args)
        self.params += list(self.qmixer_sa.parameters())

        if args.qmixer == "nmix":
            self.qmixer = Mixer(args)
        elif args.qmixer == "vdn":
            self.qmixer = VDNMixer()
        elif args.qmixer == "dmaq":
            self.qmixer = DMAQer(args)
        else:
            raise "qmixer error"
        self.params += list(self.qmixer.parameters())
        self.target_qmixer = copy.deepcopy(self.qmixer)

        self.central_mac = mac_REGISTRY[args.central_mac](scheme, args)
        self.params += list(self.central_mac.parameters())

        self.central_mixer = QMixerCentralFF(args)
        self.target_central_mixer = copy.deepcopy(self.central_mixer)
        self.params += list(self.central_mixer.parameters())

        self.optimiser = Adam(params=self.params, lr=args.lr)
        self.qr_optimiser = Adam(params=list(
            self.mixer.parameters()), lr=args.lr)

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_central_mac = copy.deepcopy(self.central_mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions

        self.first_qr_train = True

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q_i
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Calculate estimated central Q_i
        central_mac_out = []
        self.central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.central_mac.forward(batch, t=t)
            central_mac_out.append(agent_outs)
        central_mac_out = th.stack(central_mac_out, dim=1)  # Concat over time
        central_chosen_action_qvals_agents = th.gather(central_mac_out[:, :-1], dim=3, index=actions.unsqueeze(
            4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)  # Remove the last dim
        central_q_tot = self.central_mixer(
            central_chosen_action_qvals_agents, batch["state"][:, :-1])

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()
        is_max_joint_action = is_max_action.sum(dim=2) >= self.args.n_agents

        max_action_index_onehot = th.zeros(max_action_index.squeeze(
            3).shape + (self.n_actions,)).cuda(self.args.device)
        max_action_index_onehot = max_action_index_onehot.scatter_(
            3, max_action_index, 1)

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)
        target_mac_out = th.stack(target_mac_out, dim=1)
        target_mac_out[avail_actions == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(
                3).shape + (self.n_actions,)).cuda(self.args.device)
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(
                3, cur_max_actions, 1)
        else:
            raise "Use Double Q"

        # # Calculate the central Q-Values necessary for the target
        target_central_mac_out = []
        self.target_central_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_central_mac.forward(batch, t=t)
            target_central_mac_out.append(target_agent_outs)
        target_central_mac_out = th.stack(target_central_mac_out, dim=1)
        target_central_mac_out[avail_actions[:, :] == 0] = -9999999
        # Use the Qmix max actions
        target_central_max_qvals = th.gather(target_central_mac_out[:, :], 3, cur_max_actions[:, :].unsqueeze(
            4).repeat(1, 1, 1, 1, self.args.central_action_embed)).squeeze(3)
        target_central_q_tot = self.target_central_mixer(
            target_central_max_qvals, batch["state"])

        # QMIX
        if self.qmixer is not None and self.args.qmixer == "dmaq":  # pow qplex
            ans_chosen = self.qmixer(
                chosen_action_qvals, batch["state"][:, :-1], is_v=True)
            ans_adv = self.qmixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                                  max_q_i=max_action_qvals, is_v=False)
            q_tot_qmix = ans_chosen + ans_adv

        elif self.qmixer is not None:
            q_tot_qmix = self.qmixer(
                chosen_action_qvals, batch['state'][:, :-1])

        with th.no_grad():
            targets = build_td_lambda_targets(rewards, terminated, mask, target_central_q_tot,
                                              self.args.n_agents, self.args.gamma, self.args.td_lambda)

        mask = mask.expand_as(targets)

        # Qr training
        update_times = 10
        if self.first_qr_train:
            self.first_qr_train = False
            update_times = 20
        for i in range(update_times):
            q_tot, lambda_mut_deltaq = self.mixer(chosen_action_qvals.detach(), batch["state"][:, :-1], actions=actions_onehot,
                                                  max_q_i=max_action_qvals.detach())
            td_error = (q_tot - targets.detach())
            masked_td_error = td_error * mask
            loss = 0.5 * (masked_td_error ** 2).sum() / mask.sum()
            self.qr_optimiser.zero_grad()
            loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                params, self.args.grad_norm_clip)
            self.qr_optimiser.step()

        # QMIX training
        with th.no_grad():
            _, lambda_mut_deltaq = self.mixer(chosen_action_qvals.detach(), batch["state"][:, :-1], actions=actions_onehot,
                                              max_q_i=max_action_qvals.detach())
            condition = lambda_mut_deltaq.abs() < self.args.weighted_qmix_threshold
            if self.args.only_optimal_action:
                condition = is_max_joint_action
            condition = (condition * mask).to(th.bool)
            delta_weight = (self.args.weighted_qmix_weight - self.args.weighted_qmix_weight_finish) / \
                self.args.weighted_qmix_weight_anneal_time
            cur_weighted_qmix_weight = max(
                self.args.weighted_qmix_weight_finish, self.args.weighted_qmix_weight - t_env * delta_weight)
            training_weight = th.where(condition, th.ones_like(
                td_error) * cur_weighted_qmix_weight, th.ones_like(td_error) * self.args.other_actions_weight).detach()

        qmix_td_error = (q_tot_qmix - targets.detach())
        masked_qmix_td_error = qmix_td_error * mask
        if self.args.no_weighted_steps < t_env:
            loss = 0.5 * (masked_qmix_td_error ** 2 *
                          training_weight.detach()).sum() / mask.sum()
        else:
            loss = 0.5 * (masked_qmix_td_error ** 2).sum() / mask.sum()

        # central Q training
        central_td_error = (central_q_tot - targets.detach())
        central_mask = mask.expand_as(central_td_error)
        central_masked_td_error = central_td_error * central_mask
        loss += 0.5 * (central_masked_td_error ** 2).sum() / mask.sum()
        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            params, self.args.grad_norm_clip)
        optimiser.step()

        # print estimated matrix
        if self.args.env == "one_step_matrix_game" or self.args.env == "nstep_matrix_game":
            batch_size = batch.batch_size
            matrix_size = self.args.n_actions
            results = th.zeros((matrix_size, matrix_size))
            qr_results = th.zeros((matrix_size, matrix_size))
            weights = th.zeros((matrix_size, matrix_size))

            with th.no_grad():
                for i in range(results.shape[0]):
                    for j in range(results.shape[1]):
                        specific_joint_action = th.LongTensor([[[[i], [j]]]]).to(
                            device=mac_out.device).repeat(batch_size, 1, 1, 1)  # btn1
                        actions = batch['actions'][:, :-1]  # btn1
                        indexes = specific_joint_action == actions
                        indexes = indexes[:, :, 0:1, 0] * indexes[:, :, 1:2, 0]

                        qvals = q_tot_qmix[indexes]
                        results[i][j] = qvals.mean().item()

                        w = training_weight[indexes]
                        weights[i][j] = w.mean().item()

                        qvals = th.gather(
                            mac_out[:batch_size, 0:1], dim=3, index=specific_joint_action).squeeze(3)
                        cur_actions_onehot = th.zeros(specific_joint_action.squeeze(
                            3).shape + (self.n_actions,)).cuda(self.args.device)
                        cur_actions_onehot = cur_actions_onehot.scatter_(
                            3, specific_joint_action, 1)
                        q_tot, lambda_mut_deltaq = self.mixer(qvals, batch["state"][:, :1], actions=cur_actions_onehot,
                                                              max_q_i=max_action_qvals[:, :1])
                        qr_results[i][j] = q_tot.mean().item()

            th.set_printoptions(5, sci_mode=False)
            print('weights')
            print(weights)
            print('qr')
            print(qr_results)
            print('Qtot')
            print(results)
            print('Qi')
            print(mac_out[:, :1].mean(dim=(0, 1)).detach().cpu())
            th.set_printoptions(4)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat(
                "td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals *
                                 mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat(
                "target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env

        # return info
        info = {}
        return info

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        info = self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                              show_demo=show_demo, save_data=save_data)
        self._update_targets(episode_num)
        return info

    def _update_targets(self, episode_num):
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self.last_target_update_episode = episode_num
            self.target_mac.load_state(self.mac)
            self.target_central_mac.load_state(self.central_mac)
            if self.mixer is not None:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
                self.target_qmixer.load_state_dict(self.qmixer.state_dict())
                self.target_central_mixer.load_state_dict(
                    self.central_mixer.state_dict())

    def cuda(self):
        self.mac.cuda()
        self.central_mac.cuda()
        self.target_mac.cuda()
        self.target_central_mac.cuda()
        self.qmixer.cuda()
        self.target_qmixer.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
        if self.central_mixer is not None:
            self.central_mixer.cuda()
            self.target_central_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
