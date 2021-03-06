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

from model.net import MLPPolicy, CNNPolicy
from circle_world import StageWorld
from model.ppo import generate_action_no_sampling, transform_buffer


MAX_EPISODES = 5000
LASER_BEAM = 512
LASER_HIST = 3
HORIZON = 200
GAMMA = 0.99
LAMDA = 0.95
BATCH_SIZE = 512
EPOCH = 3
COEFF_ENTROPY = 5e-4
CLIP_VALUE = 0.1
NUM_ENV = 50
OBS_SIZE = 512
ACT_SIZE = 2
LEARNING_RATE = 5e-5



def enjoy(comm, env, policy, action_bound):


    if env.index == 0:
        env.reset_world()

    env.reset_test_pose()

    env.generate_test_goal_point()
    step = 1
    terminal = False
    ep_reward = 0

    obs = env.get_laser_observation()
    obs_stack = deque([obs, obs, obs])
    goal = np.asarray(env.get_local_goal())
    speed = np.asarray(env.get_self_speed())
    state = [obs_stack, goal, speed]

    while not rospy.is_shutdown():
        state_list = comm.gather(state, root=0)

        # generate actions at rank==0
        mean, scaled_action =generate_action_no_sampling(env=env, state_list=state_list,
                                               policy=policy, action_bound=action_bound)


        # execute actions
        real_action = comm.scatter(scaled_action, root=0)
        if terminal == True:
            real_action[0] = 0
        env.control_vel(real_action)
        # rate.sleep()
        rospy.sleep(0.001)
        # get informtion
        r, terminal, result = env.get_reward_and_terminate(step)
        step += 1
	
	ep_reward += r

        # get next state
        s_next = env.get_laser_observation()
        left = obs_stack.popleft()
        obs_stack.append(s_next)
        goal_next = np.asarray(env.get_local_goal())
        speed_next = np.asarray(env.get_self_speed())
        state_next = [obs_stack, goal_next, speed_next]


        state = state_next


    logger.info(Goal (%05.1f, %05.1f), Episode %05d, step %03d, %s' % \
                    (env.goal_point[0], env.goal_point[1], id + 1, step, result))
    logger_cal.info(ep_reward)




if __name__ == '__main__':

    # config log
    hostname = socket.gethostname()
    if not os.path.exists('./log/' + hostname + 'test_'):
        os.makedirs('./log/' + hostname + 'test_g')
    output_file = './log/' + hostname + 'test_g' + '/output.log'
    cal_file = './log/' + hostname + 'test_g' + '/cal.log'

    # config log
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
    ##########################################################

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    env = StageWorld(OBS_SIZE, index=rank, num_env=NUM_ENV)
    reward = None
    action_bound = [[0, -1], [1, 1]]

    if rank == 0:
        policy_path = 'policy'
        # policy = MLPPolicy(obs_size, act_size)
        policy = CNNPolicy(frames=LASER_HIST, action_space=2)
        policy.cuda()
        opt = Adam(policy.parameters(), lr=LEARNING_RATE)
        mse = nn.MSELoss()

        if not os.path.exists(policy_path):
            os.makedirs(policy_path)

        file = policy_path + '/stage2.pth'
        if os.path.exists(file):
            logger.info('####################################')
            logger.info('############Loading Model###########')
            logger.info('####################################')
            state_dict = torch.load(file)
            policy.load_state_dict(state_dict)
        else:
            logger.info ('Error: Policy File Cannot Find')
            exit()

    else:
        policy = None
        policy_path = None
        opt = None



    try:
        enjoy(comm=comm, env=env, policy=policy, action_bound=action_bound)
    except KeyboardInterrupt:
        import traceback
        traceback.print_exc()
