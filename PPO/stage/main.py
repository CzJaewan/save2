import os
import logging
import sys
import socket
import numpy as np
import rospy
import torch
import torch.nn as nn
from mpi4py import MPI

from torch.optim import Adam
from collections import deque

from model import MLPPolicy, CNNPolicy
from stage_env import StageWorld
from agent import PPO
from agent import transform_buffer

from tensorboardX import SummaryWriter

MAX_EPISODES = 50000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 128


GAMMA = 0.99
LAMDA = 0.95

BATCH_SIZE = 1024
EPOCH = 2
COEFF_ENTROPY = 5e-4
COEFF_VALUE = 20

NUM_ENV = 10
OBS_SIZE = 512
ACT_SIZE = 2

LEARNING_RATE = 5e-5


def run(comm, env,policy, tensorboard, policy_path):
    
    agent = PPO(env, GAMMA, LAMDA, COEFF_ENTROPY, COEFF_VALUE, LEARNING_RATE, policy, CONFIG)

    # rate = rospy.Rate(5)
    buff = []
    global_update = 0
    global_step = 0

    #world reset
    if env.index == 0:
        env.reset_world()


    for id in range(MAX_EPISODES):
        
        #reset
        env.reset_pose()

        env.generate_goal_point()
        terminal = False
        ep_reward = 0
        step = 1

        state = env.get_state()
        
        while not terminal and not rospy.is_shutdown():
        
            state_list = comm.gather(state, root=0)


            ## get_action
            #-------------------------------------------------------------------------
            # generate actions at rank==0
            v, a, logprob, scaled_action=agent.select_action(state_list=state_list)

            # execute actions
            real_action = comm.scatter(scaled_action, root=0)
            #-------------------------------------------------------------------------            
            
            ### step ############################################################
            
            state_next, r, terminal, result = env.step(real_action, step)

            ep_reward += r
            global_step += 1

            # add transitons in buff and update policy
            r_list = comm.gather(r, root=0)
            terminal_list = comm.gather(terminal, root=0)

            ########################################################################

            if global_step % HORIZON == 0:
                state_next_list = comm.gather(state_next, root=0)
                last_v, _, _, _ = agent.select_action(state_list=state_next_list)
            ## training
            #-------------------------------------------------------------------------
            if env.index == 0:
                
                buff.append((state_list, a, r_list, terminal_list, logprob, v))
                
                if len(buff) > HORIZON - 1:
                    s_batch, goal_batch, speed_batch, a_batch, r_batch, d_batch, l_batch, v_batch = \
                        transform_buffer(buff=buff)

                    t_batch, advs_batch = agent.generate_train_data(rewards=r_batch, values=v_batch,
                                                              last_value=last_v, dones=d_batch)
                    memory = (s_batch, goal_batch, speed_batch, a_batch, l_batch, t_batch, v_batch, r_batch, advs_batch)
                    
                    agent.update_policy(memory=memory)
                    buff = []
                    global_update += 1

                    summary.add_scalar('reward', ep_reward, id)
                    summary.add_scalar('policy_loss', agent.policy_loss, id)
                    summary.add_scalar('value_loss', agent.value_loss, id)
                    summary.add_scalar('entropy', agent.entropy, id)


            step += 1
            state = state_next

        if env.index == 0:
            if global_update != 0 and global_update % 20 == 0:
                torch.save(policy.state_dict(), policy_path + '/Stage1_{}'.format(global_update))
                logger.info('########################## model saved when update {} times#########'
                            '################'.format(global_update))
        distance = np.sqrt((env.goal_point[0] - env.init_pose[0])**2 + (env.goal_point[1]-env.init_pose[1])**2)

        logger.info('Env %02d, Goal (%05.1f, %05.1f), Episode %05d, step %03d, Reward %-5.1f, Distance %05.1f, %s' % \
                    (env.index, env.goal_point[0], env.goal_point[1], id + 1, step, ep_reward, distance, result))
        logger_cal.info(ep_reward)

if __name__ == '__main__':

    
    #############################################################
    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname):
        os.makedirs('./log/' + hostname)
    output_file = './log/' + hostname + '/output.log'
    cal_file = './log/' + hostname + '/cal.log'

    logger = logging.getLogger('mylogger')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    logger_cal = logging.getLogger('loggercal')
    logger_cal.setLevel(logging.INFO)
    cal_f_handler = logging.FileHandler(cal_file, mode='a')
    file_handler.setLevel(logging.INFO)
    logger_cal.addHandler(cal_f_handler)

    logger_cal.info("ep_reward")

    # tensorboard #############################################
    summary = SummaryWriter()
    ###########################################################

    comm = MPI.COMM_WORLD

    rank = comm.Get_rank()
    size = comm.Get_size()
    
    action_bound = [[0, -1], [1, 1]] ####
    
    CONFIG = [action_bound, EPOCH, BATCH_SIZE, HORIZON, NUM_ENV, LASER_HIST, OBS_SIZE, ACT_SIZE]

    env = StageWorld(512, index=rank, num_env=NUM_ENV)
    
    print("ENV")
    
    reward = None
    # torch.manual_seed(1)
    # np.random.seed(1)
    if rank == 0:
        policy_path = 'policy_agent_10'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        
        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage1_2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info('#####################################')
            logger.info('############Start Training###########')
            logger.info('#####################################')
    else:
        policy = None
        policy_path = None
        opt = None

    try:
        run(comm=comm, env=env, policy=policy, tensorboard = summary ,policy_path=policy_path)
    except KeyboardInterrupt:
        pass

