# valueIterationAgents.py
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        # Vk+1(s) = maximum in Sum(P(s'|s,a)[R(s,a,s') + gamma * Vk(s')]
        for k in range(self.iterations):   # Number of iterations
            V = self.values.copy()  # Use the value of the Vk-1 iteration.
            # For each state, calculate the maximum expected utility, update the result of the kth iteration to self.values
            for state in states:
                # Since values can be negative, max_value is set to min.
                max_value = -1000
                if (self.mdp.isTerminal(state)):   # No more iterations if it is a terminated state
                    continue
                # Calculate the expected utility (q_value) for each possible action, and save the maximum value.
                for action in self.mdp.getPossibleActions(state):
                    q_value = 0
                    # Get list of successor states that take the specified action in the state.
                    next_state_list = self.mdp.getTransitionStatesAndProbs(
                        state, action)
                    for next_state, next_probe in next_state_list:
                        R = self.mdp.getReward(
                            state, action, next_state)  # R(s,a,s’)
                        # Sum(P(s'|s,a)[R(s,a,s') + gamma * Vk(s')])
                        q_value += next_probe * \
                            (R + self.discount * V[next_state])
                    if q_value > max_value:
                        max_value = q_value
                # Update Vk(the Kth iteration)
                self.values[state] = round(max_value, 4)

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        q_value = 0
        # Get list of successor states that take the specified action in state.
        next_state_list = self.mdp.getTransitionStatesAndProbs(state, action)
        for next_state, next_probe in next_state_list:
            R = self.mdp.getReward(state, action, next_state)  # R(s,a,s’)
            q_value += next_probe * (R + self.discount * self.values[
                next_state])  # Sum(P(s'|s,a)[R(s,a,s') + gamma * Vk(s')])

        return q_value

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        max_value = -1000  # Since values can be negative, max_value is set to min.
        optimal_action = None
        for action in self.mdp.getPossibleActions(state):
            q_value = 0
            # Get list of successor states that take the specified action in the state state.
            next_state_list = self.mdp.getTransitionStatesAndProbs(state, action)
            for next_state, next_probe in next_state_list:
                R = self.mdp.getReward(state, action, next_state)  # R(s,a,s’)
                q_value += next_probe * (R + self.discount * self.values[next_state])  # Sum(P(s'|s,a)[R(s,a,s') + gamma * Vk(s')])
            if q_value > max_value:
                max_value = q_value
                optimal_action = action

        return optimal_action

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        state_num = len(states)

        # Vk+1(s) = maximum in Sum(P(s'|s,a)[R(s,a,s') + gamma * Vk(s')]
        for k in range(self.iterations):  # Number of iterations
            state_index = k % state_num
            state = states[state_index]
            V = self.values.copy()  # Use the value of the Vk-1 iteration.
            max_value = -1000  # Since values can be negative, max_value is set to min.
            if (self.mdp.isTerminal(state)):  # No more iterations if a terminated state
                continue
            # Calculate the expected utility (q_value) for each possible action, and save the maximum value.
            for action in self.mdp.getPossibleActions(state):
                q_value = 0
                # Get list of successor states that take the specified action in the state state.
                next_state_list = self.mdp.getTransitionStatesAndProbs(state,
                                                                       action)
                for next_state, next_probe in next_state_list:
                    R = self.mdp.getReward(state, action, next_state)  # R(s,a,s’)
                    # Sum(P(s'|s,a)[R(s,a,s') + gamma * Vk(s')])
                    q_value += next_probe * (R + self.discount * V[next_state])
                if q_value > max_value:
                    max_value = q_value
            self.values[state] = round(max_value,
                                       4)  # Update Vk(the Kth iteration)

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = {}  # save predecessors of all states
        q_values = self.values.copy()  # save max_q_value of all states
        queue = util.PriorityQueue()

        # For each state:
        #  (1) For each next state of state, save state as a predecessor
        #  (2) Find diff and push it to pq
        for state in self.mdp.getStates():
            max_value = -1000
            for action in self.mdp.getPossibleActions(state):
                next_state_list = self.mdp.getTransitionStatesAndProbs(state,
                                                                       action)
                for next_state, next_probe in next_state_list:
                    if next_probe:
                        if next_state not in states:
                            states[next_state] = {
                                state}  # state is predecessor of next_state, add it
                        else:
                            states[next_state].add(
                                state)  # state is predecessor of next_state, add it
                q_value = self.computeQValueFromValues(state, action)
                if q_value > max_value:
                    max_value = q_value
            if not self.mdp.isTerminal(state):
                q_values[state] = round(max_value,
                                        4)  # save max q_value from state across action
                diff = abs(
                    (self.values[state] - q_values[state]))  # diff is positive
                queue.push(state, -diff)  # use a negative because the priority queue is a min heap

        for k in range(self.iterations):
            if queue.isEmpty():
                break
            state = queue.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = q_values[
                    state]  # update self.values of state using max q_value
            prev_states = states[state]
            for p in prev_states:  # For each predecessor p of state,
                # find diff and push it to pq
                max_value = -1000
                for action in self.mdp.getPossibleActions(p):
                    q_value = self.computeQValueFromValues(p, action)
                    if q_value > max_value:
                        max_value = q_value
                q_values[p] = round(max_value, 4)
                diff = abs(self.values[p] - q_values[p])
                if diff > self.theta:
                    queue.push(p, -diff)


