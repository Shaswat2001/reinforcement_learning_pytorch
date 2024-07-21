import numpy as np
import torch
from common.exploration import OUActionNoise
from common.replay_buffer import ReplayBuffer
from network.ddpg_critic import DDPGCritic
from common.utils import hard_update,soft_update
import os

class TD3:

    def __init__(self,args,policy):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.learning_step = 0 
        self.policy = policy
        
        self.reset()

    def choose_action(self,state,stage = "training"):
        
        state = torch.Tensor(state).to(self.device)
        if stage == "training":
            action = self.PolicyNetwork(state).to("cpu").detach().numpy()
            action += self.noiseOBJ()
        else:
            action = self.TargetPolicyNetwork(state).to("cpu").detach().numpy()

        action = np.clip(action,self.args.min_action,self.args.max_action)

        return action

    def learn(self):
        
        self.learning_step+=1
        if self.learning_step<self.args.batch_size:
            return

        state,action,reward,next_state,done = self.replay_buffer.shuffle()

        state = torch.Tensor(state).to(self.device)
        next_state = torch.Tensor(next_state).to(self.device)
        action  = torch.Tensor(action).to(self.device)
        reward = torch.Tensor(reward).to(self.device)
        next_state = torch.Tensor(next_state).to(self.device)
        done = torch.Tensor(done).to(self.device)

        target_critic_action = self.TargetPolicyNetwork(next_state)
        Q1 = self.TargetQNetwork1(next_state,target_critic_action)
        Q2 = self.TargetQNetwork2(next_state,target_critic_action)
        y = reward + self.args.gamma*torch.minimum(Q1,Q2)
        critic_value1 = self.Qnetwork1(state,action)
        critic_loss1 = torch.mean(torch.square(y - critic_value1),dim=1)
        self.QOptimizer1.zero_grad()
        critic_loss1.mean().backward(retain_graph=True)
        self.QOptimizer1.step()

        critic_value2 = self.Qnetwork2(state,action)
        critic_loss2 = torch.mean(torch.square(y - critic_value2),dim=1)
        self.QOptimizer2.zero_grad()
        critic_loss2.mean().backward(retain_graph=True)
        self.QOptimizer2.step()

        actions = self.PolicyNetwork(state)
        critic_value = self.Qnetwork1(state,actions)
        actor_loss = -critic_value.mean()
        self.PolicyOptimizer.zero_grad()
        actor_loss.mean().backward()
        self.PolicyOptimizer.step()
        

        if self.learning_step%self.args.target_update == 0:                
            soft_update(self.TargetPolicyNetwork,self.PolicyNetwork,self.args.tau)
            soft_update(self.TargetQNetwork1,self.Qnetwork1,self.args.tau)
            soft_update(self.TargetQNetwork2,self.Qnetwork2,self.args.tau)

    def reset(self):

        self.replay_buffer = ReplayBuffer(self.args.state_size,mem_size = self.args.mem_size,n_actions = self.args.n_actions,batch_size = self.args.batch_size)
        # Exploration Technique
        self.noiseOBJ = OUActionNoise(mean=np.zeros(self.args.n_actions), std_deviation=float(0.08) * np.ones(self.args.n_actions))
        
        self.PolicyNetwork = self.policy(self.args).to(self.device)
        self.PolicyOptimizer = torch.optim.Adam(self.PolicyNetwork.parameters(),lr=self.args.actor_lr)
        self.TargetPolicyNetwork = self.policy(self.args).to(self.device)

        self.Qnetwork1 = DDPGCritic(self.args).to(self.device)
        self.QOptimizer1 = torch.optim.Adam(self.Qnetwork1.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork1 = DDPGCritic(self.args).to(self.device)

        self.Qnetwork2 = DDPGCritic(self.args).to(self.device)
        self.QOptimizer2 = torch.optim.Adam(self.Qnetwork2.parameters(),lr=self.args.critic_lr)
        self.TargetQNetwork2 = DDPGCritic(self.args).to(self.device)

        hard_update(self.TargetPolicyNetwork,self.PolicyNetwork)
        hard_update(self.TargetQNetwork1,self.Qnetwork1)
        hard_update(self.TargetQNetwork2,self.Qnetwork2)

    def add(self,s,action,rwd,next_state,done):
        self.replay_buffer.store(s,action,rwd,next_state,done)
    
    def save(self,env):
        print("-------SAVING NETWORK -------")

        os.makedirs("config/saves/training_weights/"+ env + "/td3_weights", exist_ok=True)
        torch.save(self.PolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/td3_weights/actorWeights.pth")
        torch.save(self.TargetPolicyNetwork.state_dict(),"config/saves/training_weights/"+env+"/td3_weights/TargetactorWeights.pth")

    def load(self,env):

        self.PolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/td3_weights/actorWeights.pth",map_location=torch.device('cpu')))
        self.TargetPolicyNetwork.load_state_dict(torch.load("config/saves/training_weights/"+env+"/td3_weights/TargetactorWeights.pth",map_location=torch.device('cpu')))