from . import DQN
from . import DQN_old
from . import DNN2
from .FuzzyControl import fuzzy_agent
import numpy as np
import os
import tensorflow as tf

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
#MEMORY_PATH = os.path.join(DIR_PATH, "memory/memory.npz")
MODEL_PATH = [
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule2_n7.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule3_n7.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule4_n7.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule5_n7.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule6_n7.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule2_n7_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule3_n7_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule4_n7_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule5_n7_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_memory2018_rule6_n7_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_supervised_n7.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_supervised_n7_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_supervised_n20.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_supervised_n20_NT.ckpt"),
                os.path.join(DIR_PATH, "model\\kite_CloneFLC_n7_NT.ckpt"),
                #os.path.join(DIR_PATH, "model\\kite_supervised_n30.ckpt"),
                #os.path.join(DIR_PATH, "model\\kite_supervised_n30_NT.ckpt"),
                None,
                ]
N_INPUT = [ 70, 70, 70, 70, 70, 56, 56, 56, 56, 56, 63, 49, 180, 140, 70, 49]
AI_RL   = [ 1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 ,1 , 0, 0, 0, 0, 0, 0]
AI_NT   = [ 0 ,0 ,0 ,0 ,0 , 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1]
AI_DISCRIPTION = [
                    '# RL. rule 2: point to target point (100 below center)',
                    '# RL. rule 3: close to target point (100 below center)',
                    '# RL. rule 4: rule 2 + 3',
                    '# RL. rule 5: opposite to rule 2.',
                    '# RL. rule 6: -exp(distance).',
                    '# RL. rule 2: No tension input',
                    '# RL. rule 3: No tension input',
                    '# RL. rule 4: No tension input',
                    '# RL. rule 5: No tension input',
                    '# RL. rule 6: No tension input',
                    '# DNN. input 7 records',
                    '# DNN. input 7 records, No tension input',
                    '# DNN. input 20 records',
                    '# DNN. input 20 records, No tension input',
                    '# DNN. Clone FLC. input 7 records, No tension input',
                    #'# DNN. input 30 records',
                    #'# DNN. input 30 records, No tension input',
                    '# Fuzzy logic control.',
                    ]

def build_AI(AI_MODEL=0):
    if MODEL_PATH[AI_MODEL] is None:
        AI = fuzzy_agent()
        return AI
    if AI_RL[AI_MODEL]: # Reinforcement learning AI
        AI = DQN.DeepQNetwork(n_actions=5,
                    n_input=N_INPUT[AI_MODEL],
                    learning_rate=0.0005,
                    lr_decay_step = 6000,#600,
                    lr_decay_rate = 0.93,
                    reward_decay = 0.9,
                    batch_size=1000,#10000,
                    replace_target_iter=100,
                    # e_greedy=0.9,
                    # e_greedy_increment=0.002,
                    # memory_max=50000,
                    )

        
    else:
        nn_size = [N_INPUT[AI_MODEL], 500,200,200, 100, 1]
        AI = DNN2.Deep_network(nn_size = nn_size, 
                        #learning_rate = 0.005,
                        #lr_decay_step = 200,
                        #lr_decay_rate = 0.9,
                        )
    with tf.Graph().as_default():
        AI.restore(filename = MODEL_PATH[AI_MODEL])
    return AI

if __name__ == '__main__':  
    # DEBUG: check memory data  
    #print(np.sum(RL.memory_r>0)) 
    """  
    # Training model  
    training_epochs=200000  
    for _ in range(training_epochs):  
        #if RL.train_step()>20000:  
        #    RL.reset_train_step()  
        RL.learn()  
        # Print result  
        if RL.train_step() %100 ==0:  
            print(RL.train_step(), RL.get_lr(), RL.cost)  
    RL.save(filename = './model/kite_m3.ckpt')  
    RL.plot_cost('lr_m3.png') 
    """  
    """ 
    # Test model 
    RL.restore(filename = './model/kite_m3.ckpt')  
    idx = np.where(RL.memory_r!=0)[0] 
    print(idx) 
    import random  
    idx=[random.choice(idx)]  
    print(idx)  
    #idx = np.random.choice(RL.memory_size, size=10)  
    s=RL.memory_s[idx]  
    a=RL.memory_a[idx]  
    r=RL.memory_r[idx]  
    s_=RL.memory_s_[idx]  
    print('state:{}  action: {}'.format(s,a)) 
    print('Q value: {}'.format(RL.show_q(s)))  
    print(r)  
    """

