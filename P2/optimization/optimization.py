# optimization.py
# ---------------
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
import math

import numpy as np
import itertools

import pacmanPlot
import graphicsUtils
import util


# You may add any helper functions you would like here:
# def somethingUseful():
#     return True

# helper function that checks if input is an integer.
# check if floats are integers by seeing if they are w/i
# 1e-12 of the nearest integer value
def isInt(x):
    return np.abs(x - int(round(x))) <= 1e-12


def findIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b)
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.
    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).
        If none of the constraint boundaries intersect with each other, return [].

    An intersection point is an N-dimensional point that satisfies the
    strict equality of N of the input constraints.
    This method must return the intersection points for all possible
    combinations of N constraints.

    """
    "*** YOUR CODE HERE ***"
    if not constraints:
        return []
    M = len(constraints)  # number of equations/constraints
    N = len(constraints[0][0])  # number of variables
    # print("Input: ", constraints)
    # print("M: ", M)
    # print("N: ", N)

    # First, need all permutations of choosing N equations
    # out of the M total equations
    permutations = []
    form = "0" + str(M) + "b"
    for i in range(pow(2, N) - 1, pow(2, M)):  # begin w/ first value w/ N 1s
        binary_str = format(i, form)  # turns i into binary string w/ M bits
        if binary_str.count('1') == N:
            permutations.append(binary_str)
    # print(permutations)

    # Next, go through each of the permutations of equations/constraints
    # and solve for those N equations
    res = []
    for p in permutations:
        # Get all of the A's and b's
        A_matrix = []
        b_vector = []
        for i in range(0, M):
            if p[i] != '1':
                continue
            A, b = constraints[i]
            A_matrix.append(A)
            b_vector.append(b)

        # print("A_matrix: ", A_matrix)
        # print("b_vector: ", b_vector)

        # Solve for Ax=b and add answer to res.
        # If it's not square and full rank, can't be solved; move on
        if np.linalg.matrix_rank(A_matrix) != N:
            continue
        solution = np.linalg.solve(A_matrix, b_vector)
        res.append(tuple(solution))

    # print(res)
    return res


def findFeasibleIntersections(constraints):
    """
    Given a list of linear inequality constraints, return a list all
    feasible intersection points.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

    Output: A list of N-dimensional points. Each point has the form:
        (x1, x2, ..., xN).

        If none of the lines intersect with each other, return [].
        If none of the intersections are feasible, return [].

    You will want to take advantage of your findIntersections function.

    """
    "*** YOUR CODE HERE ***"
    res = []  # feasible intersections
    intersections = findIntersections(constraints)

    # if none of the lines intersect with each other, return []
    if not intersections:
        return []

    for point in intersections:
        point_arr = list(point)

        # check if the point is feasible for all constraints
        feasible = True
        for A_row, b in constraints:
            A_row = list(A_row)
            if np.dot(A_row, point_arr) > b:  # a1*x1 + a2*x2 + ... + aN*xN
                feasible = False
                break

        # the point satisfies the constraint, append it to res
        if feasible:
            res.append(point)

    # If none of the intersections are feasible, return []; otherwise return res
    return res


def solveLP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    find a feasible point that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your findFeasibleIntersections function.

    """
    "*** YOUR CODE HERE ***"

    if not constraints or not cost:
        return None

    bestPoint, bestObjectiveVal = None, float("inf")
    feasibleIntersections = findFeasibleIntersections(constraints)
    # Quick check; is it feasible? Return None if there is no feasible solution
    if not feasibleIntersections:
        return None

    for point in feasibleIntersections:
        # calculate the value of the objective function cost^T*x
        temp = np.dot(cost, list(point))
        if temp < bestObjectiveVal:
            bestObjectiveVal = temp
            bestPoint = point

    # returns a feasible point (tuple) that minimizes the objective
    # and the corresponding objective value at that point
    return (bestPoint, bestObjectiveVal)


def wordProblemLP():
    """
    Formulate the word problem from the write-up as a linear program.
    Use your implementation of solveLP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
            ((sunscreen_amount, tantrum_amount), maximal_utility)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    ''' weight <= 50 (0.5*shampoo + 0.25*Frap <= 50)
        space <= 100 (2.5*shampoo + 2.5*Frac <= 100)
        shampoo >= 20
        Frap >= 15.5 
        7*shampoo + 4*Frap (maximize, not minimize)
    '''
    # A list of constraints.
    constraints = []
    cost = [-7, -4]  # objective function we want to MINIMIZE
    A = [[0.5, 0.25], [2.5, 2.5], [-1, 0], [0, -1]]
    b = [50, 100, -20, -15.5]

    for i in range(0, len(A)):
        A_row = A[i]
        b_row = b[i]
        # Each constraint has the form ((a1, a2, ..., aN), b)
        constraints.append((tuple(A_row), b_row))

    # return the optimal solution or None if there's no feasible solution
    solution, negcost = solveLP(constraints, cost)
    return (solution, -negcost)


def solveIP(constraints, cost):
    """
    Given a list of linear inequality constraints and a cost vector,
    use the branch and bound algorithm to find a feasible point with
    integer values that minimizes the objective.

    Input: A list of constraints. Each constraint has the form:
        ((a1, a2, ..., aN), b).
        where the N-dimensional point (x1, x2, ..., xN) is feasible
        if a1*x1 + a2*x2 + ... + aN*xN <= b for all constraints.

        A tuple of cost coefficients: (c1, c2, ..., cN) where
        [c1, c2, ..., cN]^T is the cost vector that helps the
        objective function as cost^T*x.

    Output: A tuple of an N-dimensional optimal point and the
        corresponding objective value at that point.
        One N-demensional point (x1, x2, ..., xN) which yields
        minimum value for the objective function.

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    You can take advantage of your solveLP function.

    """
    "*** YOUR CODE HERE ***"

    # M = len(constraints) # number of equations/constraints
    N = len(constraints[0][0])  # number of variables

    # Add the LP-relaxed version of IP to priority queue Q.
    # Q stores (LP constraints, sol'n, sol'n cost) and sorted by sol'n cost
    Q = util.PriorityQueue()
    LP = constraints
    result = solveLP(constraints, cost)
    if not result:
        return None
    LPsoln, LPvalue = result
    Q.push((LP, LPsoln, LPvalue), LPvalue)

    # Loop forever
    while True:
        if Q.isEmpty():
            return None
        LP, LPsoln, LPvalue = Q.pop()

        # return LPsoln if all integer-valued. Also track xi,
        # the index of first coordinate that's not an integer
        allInt = True
        xi = 0

        for i in range(0, len(LPsoln)):
            if not isInt(LPsoln[i]):
                allInt = False
                xi = i
                break
        if allInt:
            # print(LPsoln + ", " + LPvalue)
            return (LPsoln, LPvalue)

        # use xi as the coordinate to branch off of
        # need xi <= floor(xi_val) for left branch
        # need xi >= ceil(xi_val) === -xi <= -ceil(xi_val) for right
        xi_val = LPsoln[xi]
        xi_arr = [0] * N
        xi_arr[xi] = 1  # a single 1 in an empty N array
        left_constraint = (tuple(xi_arr), np.floor(xi_val))
        xi_arr[xi] = -1
        right_constraint = (tuple(xi_arr), -np.ceil(xi_val))

        # create new left and right branches
        leftLP = LP + [left_constraint]
        rightLP = LP + [right_constraint]
        # print("LP Left: ", leftLP)
        # print("LP Right: ", rightLP)

        # find sol'ns for each branch, and add feasible ones to Q
        for branch in [leftLP, rightLP]:
            result = solveLP(branch, cost)
            if result:
                LPsoln, LPvalue = result
                Q.push((branch, LPsoln, LPvalue), LPvalue)


def wordProblemIP():
    """
    Formulate the word problem in the write-up as a linear program.
    Use your implementation of solveIP to find the optimal point and
    objective function.

    Output: A tuple of optimal point and the corresponding objective
        value at that point.
        Specifically return:
        ((f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS), minimal_cost)

        Return None if there is no feasible solution.
        You may assume that if a solution exists, it will be bounded,
        i.e. not infinity.

    """
    "*** YOUR CODE HERE ***"
    ''' gates >= 15
        sorrells >= 30
        truck weight <= 30
        num trucks == 1 per provider-community pair (6 trucks total)

    '''
    # integer units of food going to
    # f_DtoG, f_DtoS, f_EtoG, f_EtoS, f_UtoG, f_UtoS

    # Weight limits
    limit_DtoG = ((1.2, 0, 0, 0, 0, 0), 30)
    limit_DtoS = ((0, 1.2, 0, 0, 0, 0), 30)
    limit_EtoG = ((0, 0, 1.3, 0, 0, 0), 30)
    limit_EtoS = ((0, 0, 0, 1.3, 0, 0), 30)
    limit_UtoG = ((0, 0, 0, 0, 1.1, 0), 30)
    limit_UtoS = ((0, 0, 0, 0, 0, 1.1), 30)

    # Gates needs 15, Sorrells needs 30
    # DtoG + EtoG + UtoG >= 15 -> -DtoG - EtoG - UtoG <= -15
    # DtoS + EtoS + UtoS >= 30 -> -DtoS - EtoS - UtoS <= -30
    requires_G = ((-1, 0, -1, 0, -1, 0), -15)
    requires_S = ((0, -1, 0, -1, 0, -1), -30)

    # Can't have negative storage! f_XtoY ≥ 0 -> -f_XtoY ≤ 0
    noneg_DtoG = ((-1, 0, 0, 0, 0, 0), 0)
    noneg_DtoS = ((0, -1, 0, 0, 0, 0), 0)
    noneg_EtoG = ((0, 0, -1, 0, 0, 0), 0)
    noneg_EtoS = ((0, 0, 0, -1, 0, 0), 0)
    noneg_UtoG = ((0, 0, 0, 0, -1, 0), 0)
    noneg_UtoS = ((0, 0, 0, 0, 0, -1), 0)

    constraints = [limit_DtoG, limit_DtoS,
                   limit_EtoG, limit_EtoS,
                   limit_UtoG, limit_UtoS,
                   requires_G, requires_S,
                   noneg_DtoG, noneg_DtoS,
                   noneg_EtoG, noneg_EtoS,
                   noneg_UtoG, noneg_UtoS, ]
    cost = [12, 20, 4, 5, 2, 1]

    return solveIP(constraints, cost)


def foodDistribution(truck_limit, W, C, T):
    """
    Given M food providers and N communities, return the integer
    number of units that each provider should send to each community
    to satisfy the constraints and minimize transportation cost.

    Input:
        truck_limit: Scalar value representing the weight limit for each truck
        W: A tuple of M values representing the weight of food per unit for each
            provider, (w1, w2, ..., wM)
        C: A tuple of N values representing the minimal amount of food units each
            community needs, (c1, c2, ..., cN)
        T: A list of M tuples, where each tuple has N values, representing the
            transportation cost to move each unit of food from provider m to
            community n:
            [ (t1,1, t1,2, ..., t1,n, ..., t1N),
              (t2,1, t2,2, ..., t2,n, ..., t2N),
              ...
              (tm,1, tm,2, ..., tm,n, ..., tmN),
              ...
              (tM,1, tM,2, ..., tM,n, ..., tMN) ]

    Output: A length-2 tuple of the optimal food amounts and the corresponding objective
            value at that point: (optimal_food, minimal_cost)
            The optimal food amounts should be a single (M*N)-dimensional tuple
            ordered as follows:
            (f1,1, f1,2, ..., f1,n, ..., f1N,
             f2,1, f2,2, ..., f2,n, ..., f2N,
             ...
             fm,1, fm,2, ..., fm,n, ..., fmN,
             ...
             fM,1, fM,2, ..., fM,n, ..., fMN)

            Return None if there is no feasible solution.
            You may assume that if a solution exists, it will be bounded,
            i.e. not infinity.

    You can take advantage of your solveIP function.

    """
    M = len(W)
    N = len(C)

    "*** YOUR CODE HERE ***"
    # Return None if there is no feasible solution
    constraints = []

    for col, amount_food in enumerate(C):
        constraint_food = [0] * (M * N)
        for row, weight_per in enumerate(W):
            index = row * M + col
            constraint_food[index] = -1

            truck_list = [0] * (M * N)
            truck_list[row * N + col] = weight_per
            constraint_truck = tuple(truck_list)

            positive_list = [0] * (M * N)
            positive_list[row * N + col] = -1
            constraint_positive = tuple(positive_list)

            constraints.append((constraint_truck, truck_limit))
            constraints.append((constraint_positive, -0))
        constraints.append((tuple(constraint_food), -amount_food))

    cost = [0] * (M * N)
    for row, tuple_row in enumerate(T):
        for col, cost_c in enumerate(tuple_row):
            cost[row * N + col] = cost_c

    return solveIP(constraints, cost)

if __name__ == "__main__":
    constraints = [((3, 2), 10), ((1, -9), 8), ((-3, 2), 40), ((-3, -1), 20)]
    inter = findIntersections(constraints)
    print(inter)
    print()
    valid = findFeasibleIntersections(constraints)
    print(valid)
    print()
    print(solveLP(constraints, (3, 5)))
    print()
    print(solveIP(constraints, (3, 5)))
    print()
    print(wordProblemIP())
