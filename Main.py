from Game import Game
n=4
m=4
game = Game(n,m,0)

#game.print()
#game.move(Game.ACTION_UP)
#game.print()

n_states =  n*m
n_actions = 4
import numpy as np
#Q-Learning
Q = np.zeros([n_states, n_actions])
print(Q.shape)

alpha = .85
gamma = .99

nb_episodes = 1000

actions_list = []
states_list  = []
cumul_reward_list = []

for i in range(nb_episodes):
    s = game.reset()
    actions = []
    states  = [s]
    cumul_reward = 0
    e=False
    while e==False: # TANT QUE état final non atteint
        l = np.random.randn(1,n_actions)
        a = np.argmax(Q[s,:]+l)
        s1, r, e, _ = game.move(a)
        Q[s,a] = Q[s,a] + alpha * (r + gamma * np.max(Q[s1,:]) - Q[s,a])
        s = s1
        states.append(s)
        actions.append(a)
        cumul_reward += r
    states_list.append(states)
    actions_list.append(actions)
    cumul_reward_list.append(cumul_reward)

print(f"Cumulative reward : {np.array(cumul_reward_list).sum()}")
last_actions = actions_list[len(actions_list)-1]

actions = {0:'UP',1:'LEFT',2:'DOWN', 3:'RIGHT'}
[print(actions[last_actions[i]]) for i in range(len(last_actions))]