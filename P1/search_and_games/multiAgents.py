# multiAgents.py
# --------------
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


# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # lists of ghost positions (x,y) and ghost scared times (N)
        newGhostPositions = successorGameState.getGhostPositions()
        newScaredTimes = newScaredTimes
        # number of ghosts
        numGhosts = len(newGhostStates)

        # scared and not scared Ghosts
        scared = []
        notScared = []
        for i in range(0, numGhosts):
            if newScaredTimes[i] == 0:
                notScared.append(i)
            else:
                scared.append(i)

        # maximize distances from the not-scared ghosts
        distancesFromNotScared = [1000]
        for index in notScared:
            ghostPosition = newGhostPositions[index]
            dist = manhattanDistance(newPos, ghostPosition)
            distancesFromNotScared.append(dist)

        # maximize distances from the not-scared ghosts
        distancesFromScared = [1000]
        for index in scared:
            ghostPosition = newGhostPositions[index]
            dist = manhattanDistance(newPos, ghostPosition)
            distancesFromScared.append(dist)

        # eat food if possible (newFood is smaller)
        currentFood = currentGameState.getFood()
        currentFoodList = currentFood.asList()
        current_food_count = len(currentFoodList)
        newFoodList = newFood.asList()
        food_count = len(newFoodList)

        # minimize distance to nearest food
        food_dist = 1000
        for food in newFoodList:
            dist = manhattanDistance(newPos, food)
            if food_dist > dist:
                food_dist = dist


        # # compare distance(pac-man and scared ghost) to distance to reach ghost
        # scareFactor = 0 # make bigger/better if we can eat the ghost
        # for index in scared:
        #     ghostPosition = newGhostPositions[index]
        #     scaredDist = newScaredTimes[index]
        #     mazeDist = manhattanDistance(newPos, ghostPosition)
        #     if mazeDist > scaredDist: # Pacman is too far from scared ghost
        #         scareFactor -= 1
        #     elif mazeDist < scaredDist: # Pacman is can reach scared ghost
        #         scareFactor += 1
        
        
        evaluation = 10000 - min(distancesFromNotScared) - 5*food_dist
        
        # if food is right next to you, EAT IT!!!
        if current_food_count != food_count:
            evaluation += 300

        # if distance from nearest ghost is w/i 3, STAY AWAY!!!
        if min(distancesFromNotScared) < 5 or min(distancesFromScared) < 3:
            evaluation -= 2000

        # if no more food left, WIN!!!
        if food_count == 0:
            evaluation += 5000

        # if power pellet, EAT IT!!!
        if len(scared) == numGhosts:
            evaluation += 500
        
        #print(evaluation)
        return max(evaluation, 0)


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent & AlphaBetaPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 7)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        # depth counts DOWN starting at self.depth
        score, action = self.maximizer(gameState, self.depth)
        return action

    def maximizer(self, state, depth):
        if self.is_leaf(state, depth):
            return self.evaluationFunction(state), None

        best_value = -10000000  # get the max value of ...
        best_action = None

        for action in state.getLegalActions(0):
            next_state = state.generateSuccessor(0, action)
            next_value, _ = self.minimizer(1, next_state, depth)

            if next_value > best_value:
                best_value = next_value
                best_action = action

        return best_value, best_action


    def minimizer(self, index, state, depth):
        # If there are no legal actions from a state to perform the min or max
        # over, the state is a leaf even if it is not at the depth specified.
        # In this case, return the value of evaluationFunction on the state.
        if self.is_leaf(state, depth):
            return self.evaluationFunction(state), None

        worst_value = 10000000  # get the absolute min value
        worst_action = None

        for action in state.getLegalActions(index):
            next_state = state.generateSuccessor(index, action)
            if index != state.getNumAgents() - 1:  # not the last ghost
                next_value, _ = self.minimizer(index+1, next_state, depth)
            else: # already at the last ghost, move to the next level
                next_value, _ = self.maximizer(next_state, depth-1)
            if next_value < worst_value: # get the smallest value
                worst_value = next_value
                worst_action = action

        return worst_value, worst_action

    def is_leaf(self, state, depth):
        #If there are no legal actions from a state to perform the min or max
        # over, the state is a leaf even if it is not at the depth specified.
        return state.isWin() or state.isLose() or (depth == 0)



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 8)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        # depth counts DOWN starting at self.depth
        score, action = self.maximizer(gameState, self.depth)
        return action

    def maximizer(self, state, depth):
        if self.is_leaf(state, depth):
            return self.evaluationFunction(state), None

        best_value = -10000000  # get the max value of ...
        best_action = None

        for action in state.getLegalActions(0):
            next_state = state.generateSuccessor(0, action)
            next_value, _ = self.chance(1, next_state, depth)

            if next_value > best_value:
                best_value = next_value
                best_action = action

        return best_value, best_action


    def chance(self, index, state, depth):
        # If there are no legal actions from a state to perform the min or max
        # over, the state is a leaf even if it is not at the depth specified.
        # In this case, return the value of evaluationFunction on the state.
        if self.is_leaf(state, depth):
            return self.evaluationFunction(state), None

        avg_value = 0  # get the average value for ghost states
        avg_action = None

        legalActions = state.getLegalActions(index)
        for action in legalActions:
            next_state = state.generateSuccessor(index, action)
            if index != state.getNumAgents() - 1:  # not the last ghost
                next_value, _ = self.chance(index+1, next_state, depth)
            else: # already at the last ghost, move to the next level
                next_value, _ = self.maximizer(next_state, depth-1)
            
            avg_value += next_value
            avg_action = action

        return avg_value/len(legalActions), avg_action

    def is_leaf(self, state, depth):
        #If there are no legal actions from a state to perform the min or max
        # over, the state is a leaf even if it is not at the depth specified.
        return state.isWin() or state.isLose() or (depth == 0)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 9).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    # Useful information you can extract from a GameState (pacman.py)
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]


    # lists of ghost positions (x,y) and ghost scared times (N)
    ghostPositions = currentGameState.getGhostPositions()
    scaredTimes = scaredTimes
    # number of ghosts
    numGhosts = len(ghostStates)

    # scared and not scared Ghosts
    scared = []
    notScared = []
    for i in range(0, numGhosts):
        if scaredTimes[i] == 0:
            notScared.append(i)
        else:
            scared.append(i)

    # maximize distances from the not-scared ghosts
    distancesFromNotScared = [1000]
    for index in notScared:
        ghostPosition = ghostPositions[index]
        dist = manhattanDistance(position, ghostPosition)
        distancesFromNotScared.append(dist)

    # maximize(?) distances from the scared ghosts
    distancesFromScared = [1000]
    for index in scared:
        ghostPosition = ghostPositions[index]
        dist = manhattanDistance(position, ghostPosition)
        distancesFromScared.append(dist)

    # eat food if possible (newFood is smaller)
    food_list = food.asList()
    food_count = len(food_list)

    # minimize distance to nearest food
    food_dist = 1000
    for food in food_list:
        dist = manhattanDistance(position, food)
        if food_dist > dist:
            food_dist = dist
    
    evaluation = 100000

    # close food distance as much as possible
    evaluation -= 5 * food_dist
    # if food is right next to you, EAT IT!!!
    evaluation -= 900 * food_count
    # if no more food left, WIN!!!
    if food_count == 0:
        evaluation += 5000
    # if power pellet, EAT IT!!!
    if len(scared) == numGhosts:
        evaluation += 500
    # if distance from nearest ghost is w/i 2, STAY AWAY!!!
    if min(distancesFromNotScared) < 2 or min(distancesFromScared) < 2:
        evaluation = 0
    #print(evaluation)

    return max(evaluation, 0)

# Abbreviation
better = betterEvaluationFunction

