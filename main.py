#!/usr/bin/env python3

import os
import pickle
import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from agent import DDPG, TD3
from network.base_net import ContinuousMLP
from common.arguments import build_parse
from myosuite.utils import gym

def train(args,env,agent):

    best_reward = -np.inf
    total_reward = []
    avg_reward_list = []

    os.makedirs("config/saves/rl_rewards/" +args.Environment, exist_ok=True)
    os.makedirs("config/saves/images/" +args.Environment, exist_ok=True)
    
    ep_len = 0

    for i in range(args.n_episodes):

        s,_ = env.reset()
        reward = 0
        
        # for _ in range(200):
        while True:
            # s = s.reshape(1,s.shape[0])
            action = agent.choose_action(s)
            # action = agent.choose_action(s)
            next_state,rwd,done,_,_ = env.step(action)
            ep_len+=1
            agent.add(s,action,rwd,next_state,done)
            agent.learn()
            reward+=rwd
            # print(next_state)
            if done:
                break
                
            s = next_state

        total_reward.append(reward)
        avg_reward = np.mean(total_reward[-40:])

        if avg_reward>best_reward and i > 10:
            best_reward=avg_reward
            if args.save_rl_weights:
                print("Weights Saved !!!")
                agent.save(args.Environment)

        print("Episode * {} * Avg Reward is ==> {}".format(i, avg_reward))
        avg_reward_list.append(avg_reward)

    if args.save_results:
        list_cont_rwd = [avg_reward_list]
        f = open("config/saves/rl_rewards/" +args.Environment + "/" + args.Algorithm + ".pkl","wb")
        pickle.dump(list_cont_rwd,f)
        f.close()
    
    plt.title(f"Reward values - {args.Algorithm}")
    plt.xlabel("Iterations")
    plt.ylabel("Reward")
    plt.plot(avg_reward_list)
    plt.show()

if __name__=="__main__":

    args = build_parse()
    
    env = gym.make('myoLegWalk')

    args.state_size = env.observation_space.shape[0]
    args.input_shape = env.observation_space.shape[0]
    args.n_actions = env.action_space.shape[0]
    args.max_action = env.action_space.high
    args.min_action = env.action_space.low

    if args.Algorithm == "DDPG":
        agent = DDPG.DDPG(args = args,policy = ContinuousMLP)
    elif args.Algorithm == "TD3":
        agent = TD3.TD3(args = args,policy = ContinuousMLP)

    train(args,env,agent)