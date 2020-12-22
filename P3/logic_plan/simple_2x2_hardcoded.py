# simple_2x2_hardcoded.py
# -----------------------
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


'''
Simple start to messing with logical planning
'''
if __name__ == '__main__' :

    import logic
    from functools import reduce

#     # Agent symbols
#     A111, A121, A211, A221 = tuple(map(logic.Expr, ('A[1,1,1]', 'A121', 'A211', 'A221')))
#     A112, A122, A212, A222 = tuple(map(logic.Expr, ('A112', 'A122', 'A212', 'A222')))
#     A113, A123, A213, A223 = tuple(map(logic.Expr, ('A113', 'A123', 'A213', 'A223')))
#     # T...
#     
#     # Action symbols
#     N1, E1, S1, W1 = tuple(map(logic.Expr, ('N1', 'E1', 'S1', 'W1')))
#     N2, E2, S2, W2 = tuple(map(logic.Expr, ('N2', 'E2', 'S2', 'W2')))
#     # T...

    # Agent symbols
    A111, A121, A211, A221 = tuple(map(logic.Expr, ('A[0,0,0]', 'A[0,1,0]', 'A[1,0,0]', 'A[1,1,0]')))
    A112, A122, A212, A222 = tuple(map(logic.Expr, ('A[0,0,1]', 'A[0,1,1]', 'A[1,0,1]', 'A[1,1,1]')))
    A113, A123, A213, A223 = tuple(map(logic.Expr, ('A[0,0,2]', 'A[0,1,2]', 'A[1,0,2]', 'A[1,1,2]')))
    # T...
    
    # Action symbols
    N1, E1, S1, W1 = tuple(map(logic.Expr, ('N[0]', 'E[0]', 'S[0]', 'W[0]')))
    N2, E2, S2, W2 = tuple(map(logic.Expr, ('N[1]', 'E[1]', 'S[1]', 'W[1]')))
    # T...

    # Axioms

    rules = []

    # At least one agent at each time
    rules += [A111 | A121 | A211 | A221]
    rules += [A112 | A122 | A212 | A222]
    rules += [A113 | A123 | A213 | A223]
    # T...
    
    # At most one agent at each time
    rules += [~A111 | ~A211, ~A111 | ~A121, ~A111 | ~A221]
    rules += [~A211 | ~A121, ~A211 | ~A221]
    rules += [~A121 | ~A221]
    # T2
    rules += [~A112 | ~A212, ~A112 | ~A122, ~A112 | ~A222]
    rules += [~A212 | ~A122, ~A212 | ~A222]
    rules += [~A122 | ~A222]
    # T3
    rules += [~A113 | ~A213, ~A113 | ~A123, ~A113 | ~A223]
    rules += [~A213 | ~A123, ~A213 | ~A223]
    rules += [~A123 | ~A223]
    # T...

#     for r in rules:
#         print(r)
#     print()
    
    # At most one action at each time
    rules += [~N1 | ~E1, ~N1 | ~S1, ~N1 | ~W1]
    rules += [~E1 | ~S1, ~E1 | ~W1]
    rules += [~S1 | ~W1]
    # T2
    rules += [~N2 | ~E2, ~N2 | ~S2, ~N2 | ~W2]
    rules += [~E2 | ~S2, ~E2 | ~W2]
    rules += [~S2 | ~W2]
    # T...

    # Movement successor state axioms
    # T1 -> T2
    rules += [A112 % ((A211 & S1) | (A121 & W1) | (A111 & ~(N1 | E1)))]
    rules += [A212 % ((A111 & N1) | (A221 & W1) | (A211 & ~(S1 | E1)))]
    rules += [A122 % ((A221 & S1) | (A111 & E1) | (A121 & ~(N1 | W1)))]
    rules += [A222 % ((A121 & N1) | (A211 & E1) | (A221 & ~(S1 | W1)))]
    # T2 -> T3
    rules += [A113 % ((A212 & S2) | (A122 & W2) | (A112 & ~(N2 | E2)))]
    rules += [A213 % ((A112 & N2) | (A222 & W2) | (A212 & ~(S2 | E2)))]
    rules += [A123 % ((A222 & S2) | (A112 & E2) | (A122 & ~(N2 | W2)))]
    rules += [A223 % ((A122 & N2) | (A212 & E2) | (A222 & ~(S2 | W2)))]
    # T...

    # BAD Effect axioms don't work. They allow for the agent to magically move even if there is no action
    # rules += [(A111 & N1) >> (A212 & ~A112), (A211 & N1) >> A212, (A121 & N1) >> (A222 & ~A122), (A221 & N1) >> A222]

    print("Rules:")
    for r in rules:
        print(r)
    print()

    init = A111
    goals = [A222, A223]

    model = False
    time = 2
    for goal in goals:
        print("Planning with t=%d" % time)
        
        s = init & reduce((lambda a,b: a&b), rules) & goal
        print(s)
    
        model = logic.dpll_satisfiable(s)
        if model :
            break
        
        time += 1

    
    print()
    if model :
        model_list = [(val,sym) for (sym, val) in model.items()]
        print(sorted(model_list, reverse=True))
    else :
        print("No can do.")
        
    
    
    
