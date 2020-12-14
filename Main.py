from Game import Game
from Grid import Grid
import numpy as np

def print_results(Q, game):
    import matplotlib.pyplot as plt
    data = Q
    fig = plt.figure()
    im = plt.imshow(data, cmap=plt.get_cmap('hot'), interpolation='nearest',
                        vmin=np.min(data), vmax=np.max(data))
    #ax.set_xticklabels(["UP","LEFT","DOWN","RIGHT"])
    #ax.set_yticklabels([str((i,j)) for i in range(game.size_x) for j in range(game.size_y)])
    #ax.grid(True)
    fig.colorbar(im)
    #ax.set_title("Q table")
    #ax.set_xlabel("Actions: UP - LEFT - DOWN - RIGHT")
    #ax.set_ylabel("Cells : ")
    plt.xlabel("actions")
    plt.ylabel("states")
    #ax.annotate('END', game.get_end_coord())
    plt.show()
    return 0

n, m = (9,9)
tp=False
if tp:
    game = Game(n,m,0)
    #game.print()
    #game.move(Game.ACTION_UP)
    #game.print()
    n_states, n_actions =  (n*m, 4)

    #Q-Learning
    Q = np.zeros([n_states, n_actions])
    print(Q.shape)

    alpha, gamma, epochs = (.85,.99, 20)
    history = {"states":[],"actions":[],"rewards":[]}
    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        s = game.reset()
        actions, states, cumul_reward = [], [s], 0
        done=False
        while done==False: # TANT QUE état final non atteint
            l = np.random.randn(1,n_actions) # noise to converge to optimal policy
            a = np.argmax(Q[s,:]+l)          # choose our best action (criterion : Q-value)
            s1, r, done, _ = game.move(a)       # do one step : (s,a) -> (s1,r) ; e = final(s1) ? True : False
            Q[s,a] = (1-alpha) * Q[s,a] + alpha * (r + gamma * np.max(Q[s1,:]) ) # Update Q-value, bootstrapping, Q-Learning
            s = s1                           # update new state
            states.append(s)
            actions.append(a)
            cumul_reward += r

        history["states"].append(states)
        history["actions"].append(actions)
        history["rewards"].append(cumul_reward)

    print(f"Cumulative reward for {epochs} epochs : {sum(history['rewards'])}")
    print(history["rewards"])
    print(len(history["rewards"]))
    print(sum(history['rewards']))
    last_actions = history["actions"][len(history["actions"])-1]
    game.print()

    actions = {0:'UP',1:'LEFT',2:'DOWN', 3:'RIGHT'}
    [print(actions[last_actions[i]]) for i in range(len(last_actions))]
elif not(tp):
    game = Grid(n,m)
    #game.print()
    #game.move(Game.ACTION_UP)
    #game.print()
    n_states, n_actions =  (n*m, 4)

    #Q-Learning
    Q = np.zeros([n_states, n_actions])
    print(Q.shape)

    alpha, gamma, epochs = (.85,.99, 1000)
    history = {"states":[],"actions":[],"rewards":[], "grids":[]}
    for epoch in range(epochs):
        print(f"{epoch}/{epochs}")
        s = game.reset()
        actions, states, cumul_reward, grids = [], [s], 0, []
        done=False
        while done==False: # TANT QUE état final non atteint
            l = np.random.randn(1,n_actions) # noise to converge to optimal policy
            a = np.argmax(Q[s,:]+l)          # choose our best action (criterion : Q-value)
            if(epoch==999):
                print(str(game.grid)+"\n")
            s1, r, done, _ = game.move(a)       # do one step : (s,a) -> (s1,r) ; e = final(s1) ? True : False
            if(epoch==999):
                print(str(game.ACTIONS_NAMES[a])+"\n"+str(game.grid)+"\n")
            Q[s,a] = (1-alpha) * Q[s,a] + alpha * (r + gamma * np.max(Q[s1,:]) ) # Update Q-value, bootstrapping, Q-Learning
            s = s1                           # update new state
            grids.append(game.grid)
            states.append(s)
            actions.append(a)
            cumul_reward += r
        if(epoch==999):
            print(f"States : {states}")
        history["grids"].append(grids)
        history["states"].append(states)
        history["actions"].append(actions)
        history["rewards"].append(cumul_reward)

    print(game.wind)
    print(f"Cumulative reward for {epochs} epochs : {sum(history['rewards'])}")
    
    print(Q.shape)
    print(f"Q matrix :\n\n {Q}")
    print_results(Q, game)
    #last_actions = history["actions"][len(history["actions"])-1]
    #[print(str(grid)+"\n"+str(action)+"\n") for (grid,action) in zip(history["grids"][-1],history["actions"][-1])]
    #game.print()
    """
    actions = {0:'UP',1:'LEFT',2:'DOWN', 3:'RIGHT'}
    [print(actions[last_actions[i]]) for i in range(len(last_actions))]
    """

