#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 04:50:53 2020

@author: hashem
"""
import numpy as np
import random

def win(state):
    
    a = 0
    outcome = False
    
    a = state[0:3] == '111'
    a += state[3:6] == '111'
    a += state[6:9] == '111'
    
    a += state[0]+state[3]+state[6] == '111'
    a += state[1]+state[4]+state[7] == '111'
    a += state[2]+state[5]+state[8] == '111'
    
    a += state[0]+state[4]+state[8] == '111'
    a += state[2]+state[4]+state[6] == '111'
    
    if not(a == 0):
        outcome = True
        
    return outcome

def lose(state):
    
    a = 0
    outcome = False
    
    a = state[0:3] == '222'
    a += state[3:6] == '222'
    a += state[6:9] == '222'
    
    a += state[0]+state[3]+state[6] == '222'
    a += state[1]+state[4]+state[7] == '222'
    a += state[2]+state[5]+state[8] == '222'
    
    a += state[0]+state[4]+state[8] == '222'
    a += state[2]+state[4]+state[6] == '222'
    
    if not(a == 0):
        outcome = True
        
    return outcome

def number_of_empty_cells(state):
    
    number_of_zeros = 0
    for i, cell in enumerate(state):
            if cell == '0':
                number_of_zeros += 1
    return number_of_zeros

def end_of_game(state):
    
    outcome = False
    if win(state) | lose(state) | (number_of_empty_cells(state) == 0):
        outcome = True
    return outcome
    
def possible_actions(state):
    
    p=[] 
    for i, cell in enumerate(state):
        if cell == '0':
            p.append(i)
    return p

def new_state(state, action, player):
    
    a = list(state)
    a[action] = str(player)
    
    return ''.join(a)

def whose_turn(game_number, state):
    
    number_of_zeros = 0
    for i, cell in enumerate(state):
        if cell == '0':
            number_of_zeros += 1
    if (game_number % 2 == 1):
        if (number_of_zeros % 2 == 1):
            return 1
        else:
            return 2
    else:
        if (number_of_zeros % 2 == 1):
            return 2
        else:
            return 1
        
def possible_new_states(state, player):
    
    possible_states = []
    possible_acts = possible_actions(state)
    for action in possible_acts:
        possible_states.append(new_state(state, action, player))
        
    return possible_states

def value(state, player):
    
    if state in all_states:
        index = all_states.index(state)
        if player ==1:
            value = values1[index]
        else:
            value = values2[index]
    else:
        value = 0   
        
    return value

def play_random(state, player):
    
    possible_acts = possible_actions(state)
    action = random.choice(possible_acts)
    
    return new_state(state, action, player)

def play_greedy(state, player):
    
    possible_states = possible_new_states(state, player)
    vals = [value(s, player) for s in possible_states]
    max_value = max(vals)
    index = [i for i, val in enumerate(vals) if val == max_value]
    if len(index) == 1:
        state_new = possible_states[index[0]]
    else:
        idx = random.choice(index)
        state_new = possible_states[idx]
    return state_new

def play_e_greedy(state, player, e=.1):
    
    if random.random() < e:
        state_new = play_random(state, player)
        greedy = False
    else:
        state_new = play_greedy(state, player)
        greedy = True
    return state_new, greedy

def play_softmax(state, player):
    
    greedy = False
    possible_states = possible_new_states(state, player)
    vals = [value(s, player) for s in possible_states]
    b = np.cumsum(np.exp(vals)/sum(np.exp(vals))) # partitioning the [0,1] proportional to softmax
    rand = random.random()
    idx = np.argmin(rand//b)
    state_new = possible_states[idx]
    if idx == np.argmax(vals):
        greedy = True
    
    return state_new, greedy
    
def update_same(states, end_state, player):
    
    reward = -1
    if player == 1:
        if win(end_state):
            reward = 10
        elif lose(end_state):
            reward = -10
    elif player == 2:
        if win(end_state):
            reward = -10
        elif lose(end_state):
            reward = 10
        
    for s in states:
        index = all_states.index(s)
        if player == 1:
            values1[index] += alpha * (reward - values1[index])
        elif player ==2:
            values2[index] += alpha * (reward - values2[index])
    return

def update_same_greedy(states, greedy_states, end_state, player):
    
    reward = -1
    if player == 1:
        if win(end_state):
            reward = 10
        elif lose(end_state):
            reward = -10
    elif player == 2:
        if win(end_state):
            reward = -10
        elif lose(end_state):
            reward = 10
        
    for s in greedy_states:
        if states.index(s) != 0:
            prev_state_idx = states.index(s)-1
            prev_state = states[prev_state_idx]
            index = all_states.index(prev_state)
            if player == 1:
                values1[index] += alpha * (reward - values1[index])
            elif player ==2:
                values2[index] += alpha * (reward - values2[index])
    return

def update_book(states, end_state, player):
    
    reward = -1
    if player == 1:
        if win(end_state):
            reward = 10
        elif lose(end_state):
            reward = -10
    elif player == 2:
        if win(end_state):
            reward = -10
        elif lose(end_state):
            reward = 10
    q = reward   
    for s in states[::-1]:
        index = all_states.index(s)
        if player == 1:
            values1[index] += alpha * (q - values1[index])
            q = values1[index]
        elif player ==2:
            values2[index] += alpha * (q - values2[index])
            q = values2[index]
    return

def update_book_greedy(states, greedy_states, end_state, player):
    
    reward = -1
    if player == 1:
        if win(end_state):
            reward = 10
        elif lose(end_state):
            reward = -10
    elif player == 2:
        if win(end_state):
            reward = -10
        elif lose(end_state):
            reward = 10
    q = reward   
    for s in greedy_states[::-1]:
        if states.index(s) != 0:
            prev_state_idx = states.index(s)-1
            prev_state = states[prev_state_idx]
            index_prev = all_states.index(prev_state)
            index_current = all_states.index(s)
            if player == 1:
                values1[index_prev] += alpha * (q - values1[index_prev])
                q = values1[index_current]
            elif player ==2:
                values2[index_prev] += alpha * (q - values2[index_prev])
                q = values2[index_current]
    return
#%%
n = 10000

alpha = .1
e = .1

player1_wins = np.zeros([16,16])
player1_losts = np.zeros([16,16])
draws = np.zeros([16,16])
# kind 0: play_random  +  update_same
# kind 1: play_random  +  update_same_greedy
# kind 2: play_random  +  update_book
# kind 3: play_random  +  update_book_greedy

# kind 4: play_greedy  +  update_same
# kind 5: play_greedy  +  update_same_greedy
# kind 6: play_greedy  +  update_book
# kind 7: play_greedy  +  update_book_greedy

# kind 8: play_e_greedy  +  update_same
# kind 9: play_e_greedy  +  update_same_greedy
# kind 10: play_e_greedy  +  update_book
# kind 11: play_e_greedy  +  update_book_greedy

# kind 12: play_softmax  +  update_same
# kind 13: play_softmax  +  update_same_greedy
# kind 14: play_softmax  +  update_book
# kind 15: play_softmax  +  update_book_greedy

for player1_kind in range(16):
    for player2_kind in range(16):
        
        game_number = 1
        state0 = '000000000'
        all_states = [state0]
        values1 = [0.0]
        values2 = [0.0]
        wins = 0
        losts = 0

        while game_number < n:
            
            states_player1_greedy = []
            states_player2_greedy = []
            states_player1 = []
            states_player2 = []
            
            state = state0
            
            while not(end_of_game(state)):
                player = whose_turn(game_number, state)
                
                if player == 1:
                    kind = player1_kind
                    
                    if kind//4 == 0:
                        state = play_random(state, player)
                        states_player1.append(state)
                        
                    elif kind//4 == 1:
                        state = play_greedy(state, player)
                        states_player1.append(state)
                        states_player1_greedy.append(state)
                        
                    elif kind//4 == 2:
                        state, greedy = play_e_greedy(state, player)
                        states_player1.append(state)
                        if greedy:
                            states_player1_greedy.append(state)
                            
                    elif kind//4 == 3:
                        state, greedy = play_softmax(state, player)
                        states_player1.append(state)
                        if greedy:
                            states_player1_greedy.append(state) 
                                 
                elif player == 2:
                    kind = player2_kind
                    
                    if kind//4 == 0:
                        state = play_random(state, player)
                        states_player2.append(state)
                        
                    elif kind//4 == 1:
                        state = play_greedy(state, player)
                        states_player2.append(state)
                        states_player2_greedy.append(state)
                        
                    elif kind//4 == 2:
                        state, greedy = play_e_greedy(state, player)
                        states_player2.append(state)
                        if greedy:
                            states_player2_greedy.append(state)
                            
                    elif kind//4 == 3:
                        state, greedy = play_softmax(state, player)
                        states_player2.append(state)
                        if greedy:
                            states_player2_greedy.append(state)   
                            
                if not(state in all_states):
                        all_states.append(state)
                        values1.append(0.0)
                        values2.append(0.0)
                        
            if player1_kind%4 == 0:
                update_same(states_player1, state, 1)
                
            elif player1_kind%4 == 1:
                update_same_greedy(states_player1, states_player1_greedy, state, 1)
                
            elif player1_kind%4 == 2:
                update_book(states_player1, state, 1)
                
            elif player1_kind%4 == 3:
                update_book_greedy(states_player1, states_player1_greedy, state, 1)
        
        
            if player2_kind%4 == 0:
                update_same(states_player2, state, 2)
                
            elif player2_kind%4 == 1:
                update_same_greedy(states_player2, states_player2_greedy, state, 2)
                
            elif player2_kind%4 == 2:
                update_book(states_player2, state, 2)
                
            elif player2_kind%4 == 3:
                update_book_greedy(states_player2, states_player2_greedy, state, 2)
        
            
            game_number += 1
            
            if game_number > .9 * n:    
                wins += win(state)
                losts += lose(state)
            
        player1_wins[player1_kind, player2_kind] = wins/(.1*n)
        player1_losts[player1_kind, player2_kind] = losts/(.1*n)
        draws[player1_kind, player2_kind] = (.1*n - wins - losts)/(.1*n)

#%%
import pandas as pd
df_wins = pd.DataFrame(player1_wins)
df_wins.to_csv('/Users/hashem/Python/RL-Sutton/player1_wins.csv')       
df_losts = pd.DataFrame(player1_losts)
df_losts.to_csv('/Users/hashem/Python/RL-Sutton/player1_losts.csv')  
df_draws = pd.DataFrame(draws)
df_draws.to_csv('/Users/hashem/Python/RL-Sutton/draws.csv')   
#%%
state0 = '000000000'
alpha = .1
e = .1
game_number = 1
wins = 0
losts = 0

while game_number < 20000:

    state = state0
    
    while not(end_of_game(state)):
        player = whose_turn(game_number, state)
        if player == 1:
            state = play_greedy(state, player)
        else:
            state = play_random(state, player)
            
    game_number += 1
    wins += win(state)
    losts += lose(state)

percentage_wins = wins/game_number
percentage_losts = losts/game_number 
percentage_draws = (game_number - wins - losts)/game_number
#%%
# def code(array):
#     h = ""
#     for i in array:
#         for j in i:
#             h = h + str(int(j))
#     return h

# flag = 1

# for l in range(100):
#     if flag == 1:
        
#         values = [0.0]
#         step = 0
#         state = np.zeros([3,3])
#         value = 0.0
#         action = [0,0]
#         flag = 0
        
#     possible_actions = []
#     possible_states = []
#     for i in np.arange(np.size(state,1)):
#         for j in np.arange(np.size(state,1)):
#             if state[i,j] == 0:
#                 possible_actions.append([i,j])
    
#     for i in possible_actions:
#         a = i[0].item()
#         b = i[1].item()
#         s = np.array(list(state))
#         s[a,b] = 1
#         possible_states.append(s)
    
#     idx = []
#     val = np.zeros(len(possible_states))
#     for k,i in enumerate(possible_states):
#         if code(i) in states:
#             index = states.index(code(i))
#             idx.append(index)
#             val[k] = (values[index])
    
#     action = possible_actions[np.argmax(val)]
#     value = np.max(val)
#     a = action[0].item()
#     b = action[1].item()
#     state[a,b] = 1
#     if win(code(state)):
#         value = 10
#         values[index] = values[index] + alpha * (value - values[index])
#     seq.append(code(state))
#     seq_val.append(value)
#     step += 1
#     if not(state in states):
#         states.append(code(state))
#         values.append(value)
        
#     if win(code(state)):
#         flag = 1
#         continue
    
#     if not(0 in state):
#         flag = 1
#         continue
#     #opponent random
#     possible_actions = []
#     for i in np.arange(np.size(state,1)):
#         for j in np.arange(np.size(state,1)):
#             if state[i,j] == 0:
#                 possible_actions.append([i,j])
                
#     opponent_action = random.choice(possible_actions)
#     a = opponent_action[0].item()
#     b = opponent_action[1].item()
#     state[a,b] = 2
    
#     if lose(code(state)):
#         value = -10
#     elif code(state) in states:
#         idx = states.index(code(state))
#         value = values[idx]
        
#     if not(code(state) in states):
#         states.append(code(state))
#         values.append(value)
#     else:
#         values[index] = values[index] + alpha * (value - values[index])
#     print(state)
#     if lose(code(state)):
#         flag = 1
#         continue
#     if not(0 in state):
#         flag = 1



















