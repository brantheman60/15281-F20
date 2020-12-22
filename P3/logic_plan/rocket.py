# rocket.py
# ---------
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


# Modified rocket planning problem from GraphPlan paper.
# Revised to include fuel measure (Claire Wang, 2019)

from graphplanUtils import *

# Types
ROCKET = 'Rocket'
PLACE = 'Place'
CARGO = 'Cargo'

# Instances
i_rocket = Instance('rocket', ROCKET)
i_london = Instance('london', PLACE)
i_paris = Instance('paris', PLACE)
i_package = Instance('package', CARGO)
i_ints = [Instance(0, INT), 
          Instance(1, INT), 
          Instance(2, INT)]

# Variables
v_from = Variable('from', PLACE)
v_to = Variable('to', PLACE)
v_place = Variable('place', PLACE)
v_cargo = Variable('cargo', CARGO)
v_fuel_start = Variable('start fuel', INT)
v_fuel_end = Variable('end fuel', INT)

# Operators
o_move = Operator('move',   # The name of the action
    # Preconditions
    [Proposition(NOT_EQUAL, v_from, v_to),
     Proposition('at', i_rocket, v_from),
     Proposition('fuel_at', v_fuel_start),
     Proposition(LESS_THAN, i_ints[0], v_fuel_start),
     Proposition(SUM, i_ints[1], v_fuel_end, v_fuel_start)],
    # Adds
    [Proposition('at', i_rocket, v_to),
     Proposition('fuel_at', v_fuel_end)],
    # Deletes
    [Proposition('at', i_rocket, v_from),
     Proposition('fuel_at', v_fuel_start)])

o_unload = Operator('unload',   # The name of the action
    # Preconditions
    [Proposition('at', i_rocket, v_place),
     Proposition('in', v_cargo, i_rocket)],
    # Adds
    [Proposition('at', v_cargo, v_place)],
    # Deletes
    [Proposition('in', v_cargo, i_rocket)])

o_load = Operator('load',   # The name of the action
    # Preconditions
    [Proposition('at', i_rocket, v_place),
     Proposition('at', v_cargo, v_place)],
    # Adds
    [Proposition('in', v_cargo, i_rocket)],
    # Deletes
    [Proposition('at', v_cargo, v_place)])


prob1 = GraphPlanProblem('little einsteins',
    # Instances
    [i_rocket, i_london, i_paris, i_package] + i_ints,
    # Operators
    [o_move, o_unload, o_load],
    # Start state
    [Proposition('at', i_package, i_london),
     Proposition('at', i_rocket, i_london),
     Proposition('fuel_at', i_ints[2])],
    # Goals
    [Proposition('at', i_package, i_paris),
     Proposition('at', i_rocket, i_london)])

# we're going on a trip, in our favorite rocket ship
# zooming through the sky...
prob1.solve()
prob1.display()

# Debugging
#prob1.dump()
