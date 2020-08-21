import torch
import logging

import os
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import socket
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

#############################################################
# config log
##############################################################
hostname = socket.gethostname()
if not os.path.exists('./log/' + hostname):
    os.makedirs('./log/' + hostname)
ppo_file = './log/' + hostname + '/ppo.log'

logger_ppo = logging.getLogger('loggerppo')
logger_ppo.setLevel(logging.INFO)
ppo_file_handler = logging.FileHandler(ppo_file, mode='a')
ppo_file_handler.setLevel(logging.INFO)
logger_ppo.addHandler(ppo_file_handler)

############################################################
logger_ppo.info("policy loss, value loss, entropy")
###########################################################

def transform_buffer(buff):
    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, \
    v_batch = [], [], [], [], [], [], [], []
    s_temp, goal_temp, speed_temp = [], [], []

    for e in buff:
        for state in e[0]:
            s_temp.append(state[0])
            goal_temp.append(state[1])
            speed_temp.append(state[2])
        s_batch.append(s_temp)
        goal_batch.append(goal_temp)
        speed_batch.append(speed_temp)
        s_temp = []
        goal_temp = []
        speed_temp = []

        a_batch.append(e[1])
        r_batch.append(e[2])
        d_batch.append(e[3])
        l_batch.append(e[4])
        v_batch.append(e[5])

    s_batch = np.asarray(s_batch)
    goal_batch = np.asarray(goal_batch)
    speed_batch = np.asarray(speed_batch)
    a_batch = np.asarray(a_batch)
    r_batch = np.asarray(r_batch)
    d_batch = np.asarray(d_batch)
    l_batch = np.asarray(l_batch)
    v_batch = np.asarray(v_batch)

    return s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch

class PPO(object):
    def __init__(self, env, gamma, lam, coeff_entropy, coeff_value, learning_rate, policy, config):
        
        self.env = env

        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.lam = lam
        self.learning_rate = learning_rate
        self.coeff_entropy = coeff_entropy
        self.coeff_value = coeff_value
        
        action_bound, epoch, batch_size, num_step, num_env, frames, obs_size, act_size = config
        
        
        self.action_bound = action_bound
        self.epoch = epoch
        self.batch_size = batch_size
        self.num_step = num_step
        self.num_env = num_env
        self.frames = frames
        self.obs_size = obs_size
        self.act_size = act_size


        self.policy_loss = 0
        self.value_loss = 0
        self.entropy = 0


    def select_action(self, state_list):
        if self.env.index == 0:
            s_list, goal_list, speed_list = [], [], []
            for i in state_list:
                s_list.append(i[0])
                goal_list.append(i[1])
                speed_list.append(i[2])

            s_list = np.asarray(s_list)
            goal_list = np.asarray(goal_list)
            speed_list = np.asarray(speed_list)

            s_list = Variable(torch.from_numpy(s_list)).float().cuda()
            goal_list = Variable(torch.from_numpy(goal_list)).float().cuda()
            speed_list = Variable(torch.from_numpy(speed_list)).float().cuda()

            v, a, logprob, mean = self.policy(s_list, goal_list, speed_list)
            v, a, logprob = v.data.cpu().numpy(), a.data.cpu().numpy(), logprob.data.cpu().numpy()
            scaled_action = np.clip(a, a_min=self.action_bound[0], a_max=self.action_bound[1])
        else:
            v = None
            a = None
            scaled_action = None
            logprob = None

        return v, a, logprob, scaled_action

    def generate_train_data(self, rewards, values, last_value, dones):
        num_step = rewards.shape[0]
        num_env = rewards.shape[1]
        values = list(values)
        values.append(last_value)
        values = np.asarray(values).reshape((num_step+1,num_env))

        targets = np.zeros((num_step, num_env))
        gae = np.zeros((num_env,))

        for t in range(num_step - 1, -1, -1):
            delta = rewards[t, :] + self.gamma * values[t + 1, :] * (1 - dones[t, :]) - values[t, :]
            gae = delta + self.gamma * self.lam * (1 - dones[t, :]) * gae

            targets[t, :] = gae + values[t, :]

        advs = targets - values[:-1, :]
        return targets, advs

    def update_policy(self, memory):
        obss, goals, speeds, actions, logprobs, targets, values, rewards, advs = memory

        advs = (advs - advs.mean()) / advs.std()

        obss = obss.reshape((self.num_step*self.num_env, self.frames, self.obs_size))
        goals = goals.reshape((self.num_step*self.num_env, 2))
        speeds = speeds.reshape((self.num_step*self.num_env, 2))
        actions = actions.reshape(self.num_step*self.num_env, self.act_size)
        logprobs = logprobs.reshape(self.num_step*self.num_env, 1)
        advs = advs.reshape(self.num_step*self.num_env, 1)
        targets = targets.reshape(self.num_step*self.num_env, 1)

        for update in range(self.epoch):
            sampler = BatchSampler(SubsetRandomSampler(list(range(advs.shape[0]))), batch_size=self.batch_size,
                                drop_last=False)
            for i, index in enumerate(sampler):
                sampled_obs = Variable(torch.from_numpy(obss[index])).float().cuda()
                sampled_goals = Variable(torch.from_numpy(goals[index])).float().cuda()
                sampled_speeds = Variable(torch.from_numpy(speeds[index])).float().cuda()

                sampled_actions = Variable(torch.from_numpy(actions[index])).float().cuda()
                sampled_logprobs = Variable(torch.from_numpy(logprobs[index])).float().cuda()
                sampled_targets = Variable(torch.from_numpy(targets[index])).float().cuda()
                sampled_advs = Variable(torch.from_numpy(advs[index])).float().cuda()

                new_value, new_logprob, dist_entropy = self.policy.evaluate_actions(sampled_obs, sampled_goals, sampled_speeds, sampled_actions)
                ratio = torch.exp(new_logprob - sampled_logprobs)

                sampled_logprobs = sampled_logprobs.view(-1, 1)
                sampled_advs = sampled_advs.view(-1, 1)
                sampled_targets = sampled_targets.view(-1, 1)

                surrogate1 = ratio * sampled_advs
                surrogate2 = torch.clamp(ratio, 1 - clip_value, 1 + clip_value) * sampled_advs

                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                
                value_loss = F.mse_loss(new_value, sampled_targets)

                loss = policy_loss + self.coeff_value * value_loss - self.coeff_entropy * dist_entropy
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                info_p_loss, info_v_loss, info_entropy = float(policy_loss.detach().cpu().numpy()), \
                                                        float(value_loss.detach().cpu().numpy()), float(
                                                        dist_entropy.detach().cpu().numpy())
                logger_ppo.info('{}, {}, {}'.format(info_p_loss, info_v_loss, info_entropy))
                
                self.policy_loss = info_p_loss
                self.value_loss = info_v_loss
                self.entropy = info_entropy

        print('update')



