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
        the_next_val = self.values.copy()
        for i in range(self.iterations):
            for cur_state in self.mdp.getStates():
                if not self.mdp.isTerminal(cur_state):
                    for_max_val = []
                    for action in self.mdp.getPossibleActions(cur_state):
                        for_max_val.append(self.getQValue(cur_state, action))
                    the_next_val[cur_state] = max(for_max_val)    
            self.values = the_next_val.copy()

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
        # successors = [(next_state, prob),]
        # successors = self.mdp.getTransitionStatesAndProbs(state, action)
        # qval = 0
        # for next_state, prob in successors:
        #     qval += prob * (self.mdp.getReward(state, action, next_state)
        #                     + self.discount * self.getValue(next_state))
        # return qval

        (calculate_qval, nexter) = (0, self.mdp.getTransitionStatesAndProbs(state, action))
        calculater = lambda x,y: y * (self.mdp.getReward(state, action, x) + self.discount * self.getValue(x))
        for next_step, block_probability in nexter:
            calculate_qval += calculater(next_step, block_probability)
        return calculate_qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        strategy = util.Counter()
        for action in self.mdp.getPossibleActions(state):
            strategy[action] = self.getQValue(state, action)
        return strategy.argMax()

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
        box_states = self.mdp.getStates()
        totalStates = len(box_states)
        for i in range(self.iterations):
            state = box_states[i % totalStates]
            if not self.mdp.isTerminal(state):
                max_Q_val = -float('inf')
                for action in self.mdp.getPossibleActions(state):
                    q_val = self.computeQValueFromValues(state, action)
                    if q_val > max_Q_val:
                        max_Q_val = q_val
                self.values[state] = max_Q_val

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
        box_states = self.mdp.getStates()
        # numStates = len(box_states)

        pre_dict = {}
        for state in [ i if self.mdp.isTerminal(i) else not self.mdp.isTerminal(i) for i in box_states]:
            for action in self.mdp.getPossibleActions(state):
                pair_box = []
                for pair in self.mdp.getTransitionStatesAndProbs(state, action):
                    pair_box.append(pair[0])
                for nextState in pair_box:
                    if nextState in pre_dict:
                        continue
                    else:    
                        pre_dict[nextState] = set()
                    pre_dict[nextState].add(state)

        youxian_queue = util.PriorityQueue()
        for state in box_states:
            if not self.mdp.isTerminal(state):
                max_q = max([self.computeQValueFromValues(state, a) for a in self.mdp.getPossibleActions(state)])
                margin = abs(max_q - self.getValue(state))
                youxian_queue.update(state, -margin)

        for i in range(self.iterations):
            if youxian_queue.isEmpty():
                break

            state = youxian_queue.pop()
            if not self.mdp.isTerminal(state):
                max_q = max([self.computeQValueFromValues(state, i) for i in self.mdp.getPossibleActions(state)])
                self.values[state] = max_q
                for pred in pre_dict[state]:
                    max_q = max([self.computeQValueFromValues(pred, i) for i in self.mdp.getPossibleActions(pred)])
                    margin = abs(max_q - self.getValue(pred))
                    if margin > self.theta:
                        youxian_queue.update(pred, -margin)

