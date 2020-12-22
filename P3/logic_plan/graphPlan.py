# graphPlan.py
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
In graphPlan.py, you will implement graph plan planning methods which are called by
Pacman agents (in graphPlanAgents.py).
"""

import util
import sys
import logic
import game
from graphplanUtils import *

OPEN = "Open"
FOOD = "Food"
PACMAN = "Pacman"


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


"""
    Create and solve the pacman navigation problem. 
    You will create instances, variables, and operators for pacman's actions 
    'North','South','East','West'
    Operators contain lists of preconditions, add effects, and delete effects
    which are all composed of propositions (boolean descriptors of the environment)
    Operators will test the current state propositions to determine whether all
    the preconditions are true, and then add and delete state propositions 
    to update the state.

    o_west = Operator('West', # the name of the action
                      [],     # preconditions
                      [],     # add effects   
                      []      # delete effects                         
                      )  

    A GraphPlan problem requires a list of all instances, all operators, the start state and 
    the goal state. You must create these lists for GraphPlan.solve.
"""


def positionGraphPlan(problem):
    width, height = problem.getWidth(), problem.getHeight()
    walls = problem.walls  # if walls[x][y] is True, then that means there is a wall at (x,y)
    walls_list = walls.asList()

    start_x = problem.startState[0]
    start_y = problem.startState[1]

    goal_x = problem.goal[0]
    goal_y = problem.goal[1]

    print('problem.startState=', problem.startState)
    print('problem.goal=', problem.goal)
    print('width=', width, 'height=', height)
    print('walls=', walls)

    """
    Create your variables with unique string names
    Each variable has a type
    vname = Variable('name',TYPE)
    TYPES = INT, PACMAN, OPEN, FOOD
    """

    # INSTANCES
    i_pacman = Instance('pacman', PACMAN)
    i_start_x, i_start_y = Instance(start_x, INT), Instance(start_y, INT)
    i_end_x, i_end_y = Instance(goal_x, INT), Instance(goal_y, INT)
    i_min_x, i_min_y = Instance(1, INT), Instance(1, INT)
    i_max_x, i_max_y = Instance(width, INT), Instance(height, INT)
    # ints from 0 to max(width,height); covers the entire x and y dimensions!
    i_ints = [Instance(i, INT) for i in range(max(width,height) + 1)]

    '''
    # create instance to store all non-wall coordinates
    # (x,y) positive if no wall at that location, negative if there is wall
    # i_open[0][0]=(0,0), i_ints[0][1]=(0,1), ...
    # from 0 to width inclusive, 0 to height inclusive

    i_opens = [[None]*(width+1) for i in range(height+1)] # height+1 x width+1 matrix
    for i in range(width):
        for j in range(height):
            if not walls[i][j]:
                i_opens.append(Instance((i, j), OPEN))
    
    
    # print("(width,height) = (%d,%d)" % (width,height))
    # print("i_open (width,height) = (%d,%d)" % (width,height))
    for x in range(width+1):
        for y in range(height+1):
            print("(x,y) = (%d,%d)" % (x,y))
            i_open[x][y] = [Instance((x,y), OPEN)]

    # i_start_open = Instance((start_x, start_y), OPEN)
    # i_end_open = Instance((goal_x, goal_y), OPEN)
    '''

    allAinstances = [i_pacman, i_start_x, i_start_y, i_end_x, i_end_y,
                     i_min_x, i_min_y, i_max_x, i_max_y
                    ] + i_ints


    # VARIABLES
    v_from_x = Variable('from_x', INT)
    v_to_x = Variable('to_x', INT)
    v_from_y = Variable('from_y', INT)
    v_to_y = Variable('to_y', INT)


    # START AND GOAL STATES
    p_open = [] # check if wall (not open) or no wall (open) at (x,y)
    for x in range(width+1):
        for y in range(height+1):
            prop = Proposition('open_at', i_ints[x], i_ints[y])
            if (x,y) in walls_list: # not open!
                p_open += [~prop]
            elif x==0 or y==0 or x==width or y==height: # not reachable!
                p_open += [~prop]
            else: # open
                p_open += [prop]

    start_state = [Proposition('pacman_at', i_pacman, i_start_x, i_start_y),
                   ] + p_open

    goal_state = [Proposition('pacman_at', i_pacman, i_end_x, i_end_y),
                  ]
    # print('start_state=', start_state)
    # print('goal_state=', goal_state)


    # OPERATORS
    current_prop = [Proposition('pacman_at', i_pacman, v_from_x, v_from_y)]
    basic_checks = [Proposition(LESS_EQUAL, i_min_y, v_from_y), # 1 ≤ y0 ≤ height
                    Proposition(LESS_EQUAL, v_from_y, i_max_y),
                    Proposition(LESS_EQUAL, i_min_x, v_from_x), # 1 ≤ x0 ≤ width
                    Proposition(LESS_EQUAL, v_from_x, i_max_x),
                    Proposition(LESS_EQUAL, i_min_y, v_to_y), # 1 ≤ y1 ≤ height
                    Proposition(LESS_EQUAL, v_to_y, i_max_y),
                    Proposition(LESS_EQUAL, i_min_x, v_to_x), # 1 ≤ x1 ≤ width
                    Proposition(LESS_EQUAL, v_to_x, i_max_x)]

    preconditions_south = [Proposition(SUM, i_ints[1], v_to_y, v_from_y), # 1 + y1 = y0
                           Proposition(EQUAL, v_from_x, v_to_x), # x1 = x0
                           Proposition('open_at', v_to_x, v_to_y),
                           ] + basic_checks + current_prop
    preconditions_west =  [Proposition(SUM, i_ints[1], v_to_x, v_from_x), # 1 + x1 = x0
                           Proposition(EQUAL, v_from_y, v_to_y), # y1 = y0
                           Proposition('open_at', v_to_x, v_to_y),
                           ] + basic_checks + current_prop
    preconditions_north = [Proposition(SUM, i_ints[1], v_from_y, v_to_y), # 1 + y0 = y1
                           Proposition(EQUAL, v_from_x, v_to_x), # x1 = x0
                           Proposition('open_at', v_to_x, v_to_y),
                           ] + basic_checks + current_prop
    preconditions_east =  [Proposition(SUM, i_ints[1], v_from_x, v_to_x), # 1 + x0 = x1
                           Proposition(EQUAL, v_from_y, v_to_y), # y1 = y0
                           Proposition('open_at', v_to_x, v_to_y),
                           ] + basic_checks + current_prop

    adds = [Proposition('pacman_at', i_pacman, v_to_x, v_to_y)]
    deletes = [Proposition('pacman_at', i_pacman, v_from_x, v_from_y)]

    o_south = Operator('South',preconditions_south, adds, deletes)
    o_west = Operator('West',preconditions_west, adds, deletes)
    o_north = Operator('North',preconditions_north, adds, deletes)
    o_east = Operator('East',preconditions_east, adds, deletes)
    alloperators = [o_south, o_west, o_north, o_east]


    # CREATE AND SOLVE THE MODEL
    prob1 = GraphPlanProblem('goto_xy', allAinstances, alloperators,
                             start_state, goal_state)
    prob1.solve()

    # some functions for debugging
    #prob1.dump()
    #prob1.display()

    actions = prob1.getactions()
    ret = []
    for action in actions:
        #print('action.print_name()=', action.print_name())
        ret.append(action.print_name())
    return ret


"""
Now use the operators for moving along with an eat operator you must create to eat 
all the food in the maze.
"""


def foodGraphPlan(problem):
    width, height = problem.getWidth(), problem.getHeight()
    walls = problem.walls  # if walls[x][y] is True, then that means there is a wall at (x,y)
    wallslist = walls.asList()

    start_x = problem.start[0][0]
    start_y = problem.start[0][1]

    foodlist = problem.start[1].asList()

    """The same as the previous question:
    Operators contain lists of preconditions, add effects, and delete effects
    which are all composed of propositions (boolean descriptors of the environment)
    Operators will test the current state propositions to determine whether all
    the preconditions are true, and then add and delete state propositions 
    to update the state.

    TYPES = INT, PACMAN, OPEN, FOOD
    """

    '''
    print('start=', start_x, start_y, width, height)
    print('walls=', walls)
    goal_x = foodlist[0][0]
    goal_y = foodlist[0][1]
    i_pacman = Instance('pacman', PACMAN)
    i_start_x = Instance(start_x, INT)
    i_start_y = Instance(start_y, INT)
    i_end_x = Instance(goal_x, INT)
    i_end_y = Instance(goal_y, INT)
    i_min_x = Instance(1, INT)
    i_min_y = Instance(1, INT)
    i_max_x = Instance(width, INT)
    i_max_y = Instance(height, INT)

    i_ends_x = []
    i_ends_y = []
    i_foods = []
    v_foods = []
    for food in foodlist:
        x = food[0]
        y = food[1]
        i_ends_x.append(Instance(x, INT))
        i_ends_y.append(Instance(y, INT))
        i_foods.append(Instance((x, y), FOOD))
        v_foods.append(Variable((x, y), FOOD))

    # create instance to store all non-wall coordinates
    i_opens = []
    for i in range(width):
        for j in range(height):
            if not walls[i][j]:
                i_opens.append(Instance((i, j), OPEN))

    i_start_open = Instance((start_x, start_y), OPEN)
    i_end_open = Instance((goal_x, goal_y), OPEN)
    i_ints = [Instance(0, INT),
              Instance(1, INT),
              Instance(2, INT)]

    for i in range(width):
        i_ints.append(Instance(i + 1, INT))
        for j in range(height):
            i_ints.append(Instance(j + 1, INT))

    allAinstances = [i_pacman, i_start_x, i_start_y, i_end_x, i_end_y,
                     i_min_x, i_min_y, i_max_x, i_max_y, i_start_open,
                     i_end_open] + i_opens + i_ints + i_ends_x + i_ends_y

    # Variables
    "*** YOUR CODE HERE ***"
    v_from_x = Variable('from_x', INT)
    v_to_x = Variable('to_x', INT)
    v_from_y = Variable('from_y', INT)
    v_to_y = Variable('to_y', INT)
    v_open = Variable('open', OPEN)
    v_open_from = Variable((v_from_x.kind, v_from_y.kind), OPEN)
    v_open_to = Variable((v_to_x.kind, v_to_y.kind), OPEN)

    start_state = [Proposition('pacman_at', i_pacman, i_start_x, i_start_y)]
    goal_state = [Proposition('pacman_at', i_pacman, i_end_x, i_end_y)]
    goal_state = []
    for i in range(len(i_ends_x)):
        goal_state.append(
            Proposition('pacman_at', i_pacman, i_ends_x[i], i_ends_y[i]))

    print('start_state=', start_state)
    print('goal_state=', goal_state)

    preconditions_south = [Proposition(LESS_EQUAL, i_min_y, v_from_y),
                           Proposition(NOT_EQUAL, v_from_y, v_to_y),
                           Proposition('pacman_at', i_pacman, v_from_x,
                                       v_from_y),
                           Proposition(SUM, i_ints[1], v_to_y, v_from_y),
                           Proposition(EQUAL, v_from_x, v_to_x)
                           # Proposition('is_open', v_open_from),
                           # Proposition(EQUAL, i_opens[0], v_open_from),
                           ]

    preconditions_west = [Proposition(LESS_EQUAL, i_min_x, v_from_x),
                          Proposition(NOT_EQUAL, v_from_x, v_to_x),
                          Proposition('pacman_at', i_pacman, v_from_x,
                                      v_from_y),
                          Proposition(SUM, i_ints[1], v_to_x, v_from_x),
                          Proposition(EQUAL, v_from_y, v_to_y)
                          # Proposition('is_open', v_open_from),
                          # Proposition(EQUAL, i_opens[0], v_open_from),
                          ]

    preconditions_north = [Proposition(LESS_EQUAL, v_from_y, i_max_y),
                           Proposition(NOT_EQUAL, v_from_y, v_to_y),
                           Proposition('pacman_at', i_pacman, v_from_x,
                                       v_from_y),
                           Proposition(SUM, i_ints[1], v_from_y, v_to_y),
                           Proposition(EQUAL, v_from_x, v_to_x)
                           # Proposition('is_open', v_open_from),
                           # Proposition(EQUAL, i_opens[0], v_open_from),
                           ]

    preconditions_east = [Proposition(LESS_EQUAL, v_from_x, i_max_x),
                          Proposition(NOT_EQUAL, v_from_x, v_to_x),
                          Proposition('pacman_at', i_pacman, v_from_x,
                                      v_from_y),
                          Proposition(SUM, i_ints[1], v_from_x, v_to_x),
                          Proposition(EQUAL, v_from_y, v_to_y)
                          # Proposition('is_open', v_open_from),
                          # Proposition(EQUAL, i_opens[0], v_open_from),
                          ]

    adds = [Proposition('pacman_at', i_pacman, v_to_x, v_to_y)
            # Proposition('is_open', v_open_to)
            ]
    deletes = [Proposition('pacman_at', i_pacman, v_from_x, v_from_y)
               # Proposition('is_open', v_open_from)
               ]

    o_south = Operator('South',
                       # Preconditions
                       preconditions_south,
                       # Adds
                       adds,
                       # Deletes
                       deletes
                       )

    o_west = Operator('West',
                      # Preconditions
                      preconditions_west,
                      # Adds
                      adds,
                      # Deletes
                      deletes
                      )

    o_north = Operator('North',
                       # Preconditions
                       preconditions_north,
                       # Adds
                       adds,
                       # Deletes
                       deletes
                       )

    o_east = Operator('East',
                      # Preconditions
                      preconditions_east,
                      # Adds
                      adds,
                      # Deletes
                      deletes
                      )
    alloperators = [o_south, o_west, o_north, o_east]
    '''

    # INSTANCES
    i_pacman = Instance('pacman', PACMAN)
    i_start_x, i_start_y = Instance(start_x, INT), Instance(start_y, INT)
    i_min_x, i_min_y = Instance(1, INT), Instance(1, INT)
    i_max_x, i_max_y = Instance(width, INT), Instance(height, INT)
    # ints from 0 to max(width,height); covers the entire x and y dimensions!
    # now, must also go from 0 to maxFood too!
    numOfFood = len(foodlist)
    i_ints = [Instance(i, INT) for i in range(max(width,height,numOfFood) + 1)]
    i_start_food = Instance(numOfFood, INT)
    i_end_food = Instance(0, INT)

    allAinstances = [i_pacman, i_start_x, i_start_y, i_start_food, i_end_food,
                     i_min_x, i_min_y, i_max_x, i_max_y
                    ] + i_ints


    # VARIABLES
    v_from_x = Variable('from_x', INT)
    v_to_x = Variable('to_x', INT)
    v_from_y = Variable('from_y', INT)
    v_to_y = Variable('to_y', INT)
    v_from_food = Variable('current food count', INT)
    v_to_food = Variable('next food count', INT)


    # START AND GOAL STATES
    p_open = [] # check if wall (not open) or no wall (open) at (x,y)
    for x in range(width+1):
        for y in range(height+1):
            prop = Proposition('open_at', i_ints[x], i_ints[y])
            if (x,y) in wallslist: # not open!
                p_open += [~prop]
            elif x==0 or y==0 or x==width or y==height: # not reachable!
                p_open += [~prop]
            else: # open
                p_open += [prop]

    p_food = [] # check if uneaten food at (x,y)
    for x in range(width+1):
        for y in range(height+1):
            prop = Proposition('food_at', i_ints[x], i_ints[y])
            if (x,y) in foodlist: # food here!
                p_food += [prop]
            elif x==0 or y==0 or x==width or y==height: # not reachable!
                p_food += [~prop]
            else: # no food here
                p_food += [~prop]


    start_state = [Proposition('pacman_at', i_pacman, i_start_x, i_start_y),
                   Proposition('food_count', i_start_food),
                   ] + p_open + p_food

    goal_state = [Proposition('food_count', i_end_food)]
    # print('start_state=', start_state)
    # print('goal_state=', goal_state)


    # OPERATORS
    current_prop = [Proposition('pacman_at', i_pacman, v_from_x, v_from_y),
                    Proposition('food_count', v_from_food)]
    basic_checks = [Proposition(LESS_EQUAL, i_min_y, v_from_y), # 1 ≤ y0 ≤ height
                    Proposition(LESS_EQUAL, v_from_y, i_max_y),
                    Proposition(LESS_EQUAL, i_min_x, v_from_x), # 1 ≤ x0 ≤ width
                    Proposition(LESS_EQUAL, v_from_x, i_max_x),
                    Proposition(LESS_EQUAL, i_min_y, v_to_y), # 1 ≤ y1 ≤ height
                    Proposition(LESS_EQUAL, v_to_y, i_max_y),
                    Proposition(LESS_EQUAL, i_min_x, v_to_x), # 1 ≤ x1 ≤ width
                    Proposition(LESS_EQUAL, v_to_x, i_max_x)]


    # PRECONDITIONS
    # ~Proposition('food_at', v_to_x, v_to_y): no food at destination
    pre_south_nf = [Proposition(SUM, i_ints[1], v_to_y, v_from_y), # 1 + y1 = y0
                    Proposition(EQUAL, v_from_x, v_to_x), # x1 = x0
                    Proposition('open_at', v_to_x, v_to_y),
                    ~Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop
    pre_west_nf =  [Proposition(SUM, i_ints[1], v_to_x, v_from_x), # 1 + x1 = x0
                    Proposition(EQUAL, v_from_y, v_to_y), # y1 = y0
                    Proposition('open_at', v_to_x, v_to_y),
                    ~Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop
    pre_north_nf = [Proposition(SUM, i_ints[1], v_from_y, v_to_y), # 1 + y0 = y1
                    Proposition(EQUAL, v_from_x, v_to_x), # x1 = x0
                    Proposition('open_at', v_to_x, v_to_y),
                    ~Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop
    pre_east_nf =  [Proposition(SUM, i_ints[1], v_from_x, v_to_x), # 1 + x0 = x1
                    Proposition(EQUAL, v_from_y, v_to_y), # y1 = y0
                    Proposition('open_at', v_to_x, v_to_y),
                    ~Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop

    adds_nf = [Proposition('pacman_at', i_pacman, v_to_x, v_to_y)]
    deletes_nf = [Proposition('pacman_at', i_pacman, v_from_x, v_from_y)]

    o_south_nf = Operator('South',pre_south_nf, adds_nf, deletes_nf)
    o_west_nf = Operator('West',pre_west_nf, adds_nf, deletes_nf)
    o_north_nf = Operator('North',pre_north_nf, adds_nf, deletes_nf)
    o_east_nf = Operator('East',pre_east_nf, adds_nf, deletes_nf)

    # new checks for if food in that direction
    food_checks = [Proposition('food_count', v_from_food), # get food count
                   Proposition(SUM, i_ints[1], v_to_food, v_from_food), # and decrease it
                   #Proposition(LESS_EQUAL, i_ints[0], v_from_food), #food count always ≥ 0
                   #Proposition(LESS_EQUAL, i_ints[0], v_to_food),
                   ]

    pre_south_f = [Proposition(SUM, i_ints[1], v_to_y, v_from_y), # 1 + y1 = y0
                    Proposition(EQUAL, v_from_x, v_to_x), # x1 = x0
                    Proposition('open_at', v_to_x, v_to_y),
                    Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop + food_checks
    pre_west_f =  [Proposition(SUM, i_ints[1], v_to_x, v_from_x), # 1 + x1 = x0
                    Proposition(EQUAL, v_from_y, v_to_y), # y1 = y0
                    Proposition('open_at', v_to_x, v_to_y),
                    Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop + food_checks
    pre_north_f = [Proposition(SUM, i_ints[1], v_from_y, v_to_y), # 1 + y0 = y1
                    Proposition(EQUAL, v_from_x, v_to_x), # x1 = x0
                    Proposition('open_at', v_to_x, v_to_y),
                    Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop + food_checks
    pre_east_f =  [Proposition(SUM, i_ints[1], v_from_x, v_to_x), # 1 + x0 = x1
                    Proposition(EQUAL, v_from_y, v_to_y), # y1 = y0
                    Proposition('open_at', v_to_x, v_to_y),
                    Proposition('food_at', v_to_x, v_to_y)
                    ] + basic_checks + current_prop + food_checks

    
    adds_f = [Proposition('pacman_at', i_pacman, v_to_x, v_to_y),
              Proposition('food_count', v_to_food), # decreased food count
              ~Proposition('food_at', v_to_x, v_to_y) # no longer food here
              ]
    deletes_f = [Proposition('pacman_at', i_pacman, v_from_x, v_from_y),
                 Proposition('food_count', v_from_food), # no longer old food count
                 Proposition('food_at', v_to_x, v_to_y) # no longer food here
                ]

    o_south_f = Operator('South',pre_south_f, adds_f, deletes_f)
    o_west_f = Operator('West',pre_west_f, adds_f, deletes_f)
    o_north_f = Operator('North',pre_north_f, adds_f, deletes_f)
    o_east_f = Operator('East',pre_east_f, adds_f, deletes_f)



    alloperators = [o_south_nf, o_west_nf, o_north_nf, o_east_nf,
                   o_south_f, o_west_f, o_north_f, o_east_f]


    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()

    prob1 = GraphPlanProblem('goto_xy', allAinstances, alloperators,
                             start_state, goal_state)
    prob1.solve()

    # some functions for debugging
    #prob1.dump()
    #prob1.display()

    actions = prob1.getactions()
    ret = []
    for action in actions:
        ret.append(action.print_name())
    "*** YOUR CODE HERE ***"
    print('ret=', ret)
    return ret


# Abbreviations
pgp = positionGraphPlan
fgp = foodGraphPlan

# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)
