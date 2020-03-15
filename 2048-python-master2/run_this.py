from puzzle import *
from RL_brain import DeepQNetwork
import pandas as pd


def run_game():

    print("\n\nrun_this run game\n\n")
    step = 0
    cost=[]
    win_count = 0
    win_rate = []
    score=[]
    temp_score=[]
    temp_cost=[]
    #先运行100次，最为初始记忆存储
    for episode in range(30):
        # initial observation
        # observation = env.reset()
        every_cost=[]
        observation = env.init_matrix()
        #******************************************************************************
        env.update_grid_cells()
        ep_score=0
        #print('\n\n\n\n')
        #print("run_this episode episode,observation",episode,observation)
        while True:
            env.render()
            # fresh env在tk中是这样，但是在
            #env.update_idletasks()
            #print("had_render")
            # RL choose action based on observation
            action = RL.choose_action(observation)
            #print("run_this episode action",action)
            # RL take action and get next observation and reward
            observation_,  done,reward,is_end,w_count= env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            ep_score+=reward
            if (step > 200) and (step % 5 == 0):
                t=RL.learn()
                every_cost.append(t)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            #if done:
            if w_count==1:
                win_count+=1

            if is_end:
                # cost.append(temp_cost[-1])
                break
            step += 1
        """key=env.key_b_press(event)
        if key=='b':
            break
"""
        temp_score.append(ep_score)
        if every_cost:
            temp_cost.append(every_cost[-1])
        if (episode + 1) % 1000 == 0:
            t=sum(temp_score)/len(temp_score)
            score.append(t)
            temp_score=[]

        if (episode + 1) % 1000 == 0:
            t = sum(temp_cost) / len(temp_cost)
            cost.append(t)
            temp_cost = []

        if (episode + 1) % 1000 == 0:
            wrate = win_count / 1000
            # print(wrate)
            win_rate.append(wrate)
            win_count = 0

    # end of game
    print('\ngame over\n')
    RL.store_cost(cost)
    RL.store_win_rate(win_rate)
    result_cost=pd.DataFrame(data=cost)
    result_cost.to_csv('D:/result2/result_cost.csv')
    result_winrate = pd.DataFrame(data=win_rate)
    result_winrate.to_csv('D:/result2/result_winrate.csv')
    result_score = pd.DataFrame(data=score)
    result_score.to_csv('D:/result2/result_score.csv')
    # print(win_rate)
    env.destroy()
    # RL.plot_cost(cost)  # 可视化cost

if __name__ == "__main__":
    # maze game
    env = GameGrid()
    print("build GameGrid")
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.001,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=500,
                      #output_graph=True
                      )
    env.after(100, run_game())
    #env.master.bind("<Key>", env.key_b_press)
    env.mainloop()
    RL.plot_cost()#可视化cost