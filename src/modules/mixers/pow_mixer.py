import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .dmaq_si_weight import DMAQ_SI_Weight

class LambdaHypernet(nn.Module):
    def __init__(self, args):
        super(LambdaHypernet, self).__init__()
        self.args = args
        self.n_agents = self.args.n_agents
        self.n_actions = self.args.n_actions
        self.state_dim = int(np.prod(self.args.state_shape))
        self.input_dim = self.state_dim + self.n_agents * self.n_actions

        self.abs = self.args.lambda_weights_abs
        hypernet_embed = self.args.lambda_hypernet_embed_dim
        self.embed_dim = self.args.lambda_mixing_embed_dim
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.input_dim, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, hypernet_embed),
                                        nn.ReLU(inplace=True),
                                        nn.Linear(hypernet_embed, self.embed_dim))

    def forward(self, inputs, states, actions):
        inputs = inputs.reshape(-1, 1, self.n_agents)
        states_actions = th.cat([states, actions], dim=1)
        # First layer
        w1 = self.hyper_w_1(states_actions).abs() if self.abs else self.hyper_w_1(states_actions)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        hidden = th.bmm(inputs, w1)
        
        # Second layer
        w_final = self.hyper_w_final(states_actions).abs() if self.abs else self.hyper_w_final(states_actions)
        w_final = w_final.view(-1, self.embed_dim, 1)
        # Compute final output
        y = th.bmm(hidden, w_final)
        y = y.reshape(-1, 1)
        return y

class PowMixer(nn.Module):
    def __init__(self, args):
        super(PowMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.state_dim = int(np.prod(args.state_shape))
        self.action_dim = self.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1

        self.embed_dim = self.args.v_embed_dim
        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

        self.si_weight = DMAQ_SI_Weight(args)
        self.lambda_hypernet = LambdaHypernet(args)

    def forward(self, agent_qs, states, actions, max_q_i):
        bs = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.reshape(-1, self.n_agents)
        actions = actions.reshape(-1, self.action_dim)
        max_q_i = max_q_i.reshape(-1, self.n_agents)

        # calc_adv
        adv_q = (agent_qs - max_q_i).view(-1, self.n_agents).detach()
        if self.args.use_lambda_hypernetwork:
            lambda_mut_deltaq = self.lambda_hypernet(adv_q, states, actions)
            adv_tot = lambda_mut_deltaq
        else:
            adv_w_final = self.si_weight(states, actions)
            adv_w_final = adv_w_final.view(-1, self.n_agents)
            lambda_mut_deltaq = th.sum(adv_q * adv_w_final, dim=1)
            adv_tot = lambda_mut_deltaq
        adv_tot = adv_tot.reshape(bs, -1, 1)
        lambda_mut_deltaq = lambda_mut_deltaq.reshape(bs, -1, 1)
        # calc v_tot
        v = self.V(states).reshape(bs, -1, 1)
        Qr = v + adv_tot
        return Qr, lambda_mut_deltaq