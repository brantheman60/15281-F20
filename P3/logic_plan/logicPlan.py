# logicPlan.py
# ------------
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


"""
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

import util
import sys
import logic
import game

pacman_str = 'P'
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'


class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()

    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()


def tinyMazePlan(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def sentence1():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.

    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** YOUR CODE HERE ***"

    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')

    ret = logic.conjoin([(A | B), (~A % (~B | C)), logic.disjoin([~A, ~B, C])])
    return ret
    util.raiseNotDefined()


def sentence2():
    """Returns a logic.Expr instance that encodes that the following expressions are all true.

    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** YOUR CODE HERE ***"

    A = logic.Expr('A')
    B = logic.Expr('B')
    C = logic.Expr('C')
    D = logic.Expr('D')

    ret = logic.conjoin(
        [(C % (B | D)), (A >> (~B & ~D)), (~(B & ~C) >> A), (~D >> C)])
    return ret
    util.raiseNotDefined()


def sentence3():
    """Using the symbols WumpusAlive[1], WumpusAlive[0], WumpusBorn[0], and WumpusKilled[0],
    created using the logic.PropSymbolExpr constructor, return a logic.PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    The Wumpus is alive at time 1 if and only if the Wumpus was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    The Wumpus cannot both be alive at time 0 and be born at time 0.

    The Wumpus is born at time 0.
    """
    "*** YOUR CODE HERE ***"
    WumpusAlive0 = logic.PropSymbolExpr('WumpusAlive', 0)
    WumpusAlive1 = logic.PropSymbolExpr('WumpusAlive', 1)

    WumpusBorn0 = logic.PropSymbolExpr('WumpusBorn', 0)
    WumpusKilled0 = logic.PropSymbolExpr('WumpusKilled', 0)

    expr1 = (
                WumpusAlive0 & ~WumpusKilled0)  # Wumpus was alive at time 0 and it was not killed at time 0
    expr2 = (
                ~WumpusAlive0 & WumpusBorn0)  # it was not alive at time 0 and it was born at time 0
    expr3 = ((WumpusAlive1) % ((expr1) | (expr2)))
    expr4 = ~(
                WumpusAlive0 & WumpusBorn0)  # cannot both be alive at time 0 and be born at time 0
    expr5 = WumpusBorn0  # The Wumpus is born at time 0

    ret = logic.conjoin([expr3, expr4, expr5])
    return ret

    util.raiseNotDefined()


def modelToString(model):
    """Converts the model to a string for printing purposes. The keys of a model are
    sorted before converting the model to a string.

    model: Either a boolean False or a dictionary of Expr symbols (keys)
    and a corresponding assignment of True or False (values). This model is the output of
    a call to logic.pycoSAT.
    """
    if model == False:
        return "False"
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def findModel(sentence):
    """Given a propositional logic sentence (i.e. a logic.Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """
    "*** YOUR CODE HERE ***"

    CNF = logic.to_cnf(sentence)
    return logic.pycoSAT(CNF)


def atLeastOne(literals):
    """
    Given a list of logic.Expr literals (i.e. in the form A or ~A), return a single
    logic.Expr instance in CNF (conjunctive normal form) that represents the logic
    that at least one of the literals in the list is true.
    >>> A = logic.PropSymbolExpr('A');
    >>> B = logic.PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(logic.pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(logic.pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(logic.pl_true(atleast1,model2))
    True
    """
    "*** YOUR CODE HERE ***"

    '''
    # Not in CNF form!
    ret = None
    for expr in literals:
        if not ret:
            ret = expr
        else:
            ret |= expr

    return ret
    '''
    # Easiest to form A v B ... v Z using logic.disjoin; this is still CNF (1 clause)
    return logic.disjoin(literals)

def atMostOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form) that represents the logic that at most one of
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"
    '''
    # Not in CNF form!
    c1 = None
    ret = None
    for index, expr in enumerate(literals):
        if not c1:
            c1 = expr
        else:
            c1 = logic.disjoin(c1, expr)

        c2 = expr
        for index2, other in enumerate(literals):
            if index != index2:
                c2 = logic.disjoin(c2, other)
        c3 = c2
        for index2, other in enumerate(literals):
            if index != index2:
                c3 = logic.conjoin(c3, ~other)

        c2_and_c3 = c3
        if not ret:
            ret = c2_and_c3
        else:
            ret = logic.disjoin(ret, c2_and_c3)

    ret = logic.disjoin(ret, ~c1)

    return logic.to_cnf(ret)
    '''
    # Ensure that no two literals are both true
    # Easily do this by looping through all pairs (A,B) A≠B
    # and showing that neg(A and B) aka ~A v ~B
    # (~A v ~B1) ^ (~A v ~B1) ^ ...
    clauses = []
    for A in literals:
        for B in literals:
            if A != B:
                clauseAB = logic.disjoin(~A, ~B)
                clauses.append(clauseAB)

    return logic.conjoin(clauses)


def exactlyOne(literals):
    """
    Given a list of logic.Expr literals, return a single logic.Expr instance in
    CNF (conjunctive normal form)that represents the logic that exactly one of
    the expressions in the list is true.
    """
    "*** YOUR CODE HERE ***"

    '''
    # Seems simple, but isn't in CNF
    return logic.conjoin(atLeastOne(literals), atMostOne(literals))
    '''

    # Ensure that 1 literal is true, but not 2
    # In DNF, it's simply (A ^ ~B ^ ~C ^ ...) v (~A ^ B ^ ~C ^ ...)
    # In CNF, first check that no 2 literals are both true (same as atMostOne).
    clauses = []
    for A in literals:
        for B in literals:
            if A != B:
                clauseAB = logic.disjoin(~A, ~B)
                clauses.append(clauseAB)

    # finally add the condition that at least 1 literal is true!
    clauses.append(logic.disjoin(literals))

    return logic.conjoin(clauses)


def extractActionSequence(model, actions):
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    "*** YOUR CODE HERE ***"
    plan = [] # List of all possible actions+values
    for symbol in model.keys():
        value = model[symbol]
        (action, t) = logic.parseExpr(symbol)
        if action in actions and value is True:
            plan.append((action, int(t)))

    plan = sorted(plan, key=lambda t: t[1]) # sort plan w/ value=True first
    action_plan = [] # List of all possible actions
    for i in range(0, len(plan)):
        (action, t) = plan[i]
        action_plan.append(action)

    return action_plan


def pacmanSuccessorStateAxioms(x, y, t, walls_grid):
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    "*** YOUR CODE HERE ***"

    P = logic.PropSymbolExpr(pacman_str, x, y, t) # original position

    # Immediately previous possible positions
    P_left = logic.PropSymbolExpr(pacman_str, x - 1, y, t - 1)
    P_right = logic.PropSymbolExpr(pacman_str, x + 1, y, t - 1)
    P_up = logic.PropSymbolExpr(pacman_str, x, y + 1, t - 1)
    P_down = logic.PropSymbolExpr(pacman_str, x, y - 1, t - 1)

    # Directions that the immediately previous position could
    # have moved to.
    # TRUE = could have moved 'East' (for instance)
    # FALSE = couldn't have moved 'East' b/c a wall in that direction
    wall_left = logic.PropSymbolExpr('East', t - 1)
    wall_right = logic.PropSymbolExpr('West', t - 1)
    wall_up = logic.PropSymbolExpr('South', t - 1)
    wall_down = logic.PropSymbolExpr('North', t - 1)

    x_max = walls_grid.width - 1
    y_max = walls_grid.height - 1

    neighbors = []
    if x > 0     and not walls_grid[x - 1][y]:
        neighbors.append(logic.conjoin(P_left, wall_left))
    if x < x_max and not walls_grid[x + 1][y]:
        neighbors.append(logic.conjoin(P_right, wall_right))
    if y < y_max and not walls_grid[x][y + 1]:
        neighbors.append(logic.conjoin(P_up, wall_up))
    if y > 0     and not walls_grid[x][y - 1]:
        neighbors.append(logic.conjoin(P_down, wall_down))

    # Ensure that it's possible for any of the previous
    # positions (N,S,E,W) to have reached current location
    ret1 = atLeastOne(neighbors)

    # Assert biconditional; these must refer to the same logic
    # that pacman can't be at P if previous positions didn't allow
    # it, and vice versa
    ret = ret1 % P

    return ret


def positionLogicPlan(problem):
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    
    MAX_TIME_STEP = 50
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    actions = ['North', 'East', 'South', 'West']
    
    "*** YOUR CODE HERE ***"

    # Continually add to these arrays of expressions as t increases
    expr_init = [] # check initial state (t=0)
    expr_prev = [] # check neighboring previous state
    expr_act = [] # check exactly 1 action was taken
    expr_goal = [] # check if current or past state is at goal

    # First add the conditions where the P(x,y,0) is not the initial state
    for x in range(1, width+1):
        for y in range(1, height+1):
            symbol = logic.PropSymbolExpr(pacman_str, x, y, 0)
            if(x, y) != (x0, y0):
                expr_init += [~symbol]
            else:
                expr_init += [symbol]

    # Loop through every neighboring position until MAX_TIME_STEP
    for t in range(MAX_TIME_STEP):
        # only for t>0, when previous positions can be found
        if t > 0:
            # call pacmanSuccessorStateAxioms on all positions that compares all
            # possible P(x,y,t-1) to P(x,y,t) to ensure they don't hit a wall
            for x in range(1, width+1):
                for y in range(1, height+1):
                    if not walls[x][y]:
                        expr_prev += [pacmanSuccessorStateAxioms(x, y, t, walls)]

            # exactly 1 action (N,S,E,W) could have been taken from previous position,
            # no more and no less!
            ex = []
            for action in actions:
                ex += [logic.PropSymbolExpr(action, t - 1)]
            expr_act += [exactlyOne(ex)]

        # finally, check if the current state is at the desired goal position
        expr_goal += [logic.PropSymbolExpr(pacman_str, xg, yg, t)]

        # time to combine all of our expressions and solve the model!
        # we DISJOIN expr_goal b/c complete if AT ANY TIME Pacman reached the
        # goal position
        clauses = expr_init + expr_prev + expr_act + [logic.disjoin(expr_goal)]
        CNF = logic.to_cnf(logic.conjoin(clauses))
        model = findModel(CNF)
        if(model is not False):
            return extractActionSequence(model, actions)

    return False


def foodLogicPlan(problem):
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    
    MAX_TIME_STEP = 50
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()
    # Note: there's no longer any goal position!
    actions = ['North', 'East', 'South', 'West']

    "*** YOUR CODE HERE ***"
    # Make a copy-paste of the previous function positionLogicPlan()
    
    # Continually add to these arrays of expressions as t increases
    expr_init = [] # check initial state (t=0)
    expr_prev = [] # check neighboring previous state
    expr_act = [] # check exactly 1 action was taken
    expr_goal = [] # check if current and past states have collectively
                   # been to every food position

    # First add the conditions where the P(x,y,0) is not the initial state
    for x in range(1, width+1):
        for y in range(1, height+1):
            symbol = logic.PropSymbolExpr(pacman_str, x, y, 0)
            if(x, y) != (x0, y0):
                expr_init += [~symbol]
            else:
                expr_init += [symbol]

    # Loop through every neighboring position until MAX_TIME_STEP
    for t in range(MAX_TIME_STEP):
        # only for t>0, when previous positions can be found
        if t > 0:
            # call pacmanSuccessorStateAxioms on all positions that compares all
            # possible P(x,y,t-1) to P(x,y,t) to ensure they don't hit a wall
            for x in range(1, width+1):
                for y in range(1, height+1):
                    if not walls[x][y]:
                        expr_prev += [pacmanSuccessorStateAxioms(x, y, t, walls)]

            # exactly 1 action (N,S,E,W) could have been taken from previous position,
            # no more and no less!
            ex = []
            for action in actions:
                ex += [logic.PropSymbolExpr(action, t - 1)]
            expr_act += [exactlyOne(ex)]

        # finally, check if for every initial food position, Pacman has been at
        # that position currently or in the past
        expr_goal = []
        for (x,y) in food:
            aux = []
            for k in range(t+1):
                aux += [logic.PropSymbolExpr(pacman_str, x, y, k)]
            expr_goal += [logic.disjoin(aux)]

        # time to combine all of our expressions and solve the model!
        # we CONJOIN expr_goal now b/c need EVERY food position to have
        # been reached at some time
        clauses = expr_init + expr_prev + expr_act + expr_goal
        CNF = logic.to_cnf(logic.conjoin(clauses))
        model = findModel(CNF)
        if(model is not False):
            return extractActionSequence(model, actions)

    return False
    


# Abbreviations
plp = positionLogicPlan
flp = foodLogicPlan

# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

