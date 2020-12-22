# simple_2x2.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import logic
import logicPlan
from functools import reduce

def move_successor_state(sym_str, x, y, t, valid_action_strs):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the valid actions at state t.
    Current <==> (causes to stay) | (causes of current)
    A[x,y,t] <==> (A[x,y,t-1] & ~(any valid action)) |
                    (A[x,y+1,t-1] & S[t-1]) |  # If N is a valid action
                    (A[x,y-1,t-1] & N[t-1]) |  # If S is a valid action
                    (A[x+1,y,t-1] & W[t-1]) |  # If E is a valid action
                    (A[x-1,y,t-1] & E[t-1]) |  # If W is a valid action
    """
    
    change_list = []
    if 'N' in valid_action_strs:
        change_list += [logic.PropSymbolExpr(sym_str,x,y+1,t-1) & logic.PropSymbolExpr('S',t-1)]
    if 'S' in valid_action_strs:
        change_list += [logic.PropSymbolExpr(sym_str,x,y-1,t-1) & logic.PropSymbolExpr('N',t-1)]
    if 'E' in valid_action_strs:
        change_list += [logic.PropSymbolExpr(sym_str,x+1,y,t-1) & logic.PropSymbolExpr('W',t-1)]
    if 'W' in valid_action_strs:
        change_list += [logic.PropSymbolExpr(sym_str,x-1,y,t-1) & logic.PropSymbolExpr('E',t-1)]
    change_exp = reduce((lambda a,b: a|b), change_list)
        
    stay_list = [~reduce((lambda a,b: a|b), logicPlan.expression_list(valid_action_strs,[t-1]))]
#     stay_list += [logic.PropSymbolExpr('G',x,y,t-1)]
    stay_exp = logic.PropSymbolExpr(sym_str,x,y,t-1) & (reduce((lambda a,b: a|b), stay_list))
    
    # Successor state axiom
    # Current <==> (causes of current) V (causes to stay)
    return logic.PropSymbolExpr(sym_str,x,y,t) % (change_exp | stay_exp)

'''
Simple start to messing with logical planning
'''
if __name__ == '__main__' :

    width = 2
    height = 2
    max_time = 3

    agent_str = 'P'
    action_strs = ['N','E','S','W']

    # Axioms

    rules = []

    # Exactly one agent at each time
    for t in range(max_time) :
        agent_positions_t = logicPlan.expression_list(agent_str,range(0,width),range(0,height),[t])
        rules += [logicPlan.exactlyOne(agent_positions_t)]
    
    # At most one action at each time
    for t in range(max_time-1):
        actions_t = logicPlan.expression_list(action_strs,[t])
        rules += [logicPlan.atMostOne(actions_t)]

    # Movement successor state axioms
    for t in range(1,max_time):
        rules += [move_successor_state(agent_str,0,0,t,['N','E'])]
        rules += [move_successor_state(agent_str,1,0,t,['N','W'])]
        rules += [move_successor_state(agent_str,0,1,t,['S','E'])]
        rules += [move_successor_state(agent_str,1,1,t,['S','W'])]
        
    print("Rules:")
    for r in rules:
        print(r)
    print()

    init = logic.PropSymbolExpr(agent_str,1,1,0)
    goals = []
    for t in range(max_time):
        goals += [logic.PropSymbolExpr(agent_str,0,0,t)]

    model = False
    for t in range(max_time):
        print("Planning with t=%d" % t)
        
        s = init & reduce((lambda a,b: a&b), rules) & goals[t]
        print(s)
    
        model = logic.dpll_satisfiable(s)
        if model :
            break
        
    
    print()
    if model:
        model_list = [(val,sym) for (sym, val) in model.items()]
        print(sorted(model_list, reverse=True))
    else:
        print("No can do.")
        
        
