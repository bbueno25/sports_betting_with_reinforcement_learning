"""
Value Iteration for Sports Betting

Value iteration helps generate v* (optimal value function) and pi* (optimal policy function).
"""

class SportsBetting:
    """
    DOCSTRING
    """
    def __init__(self):
        self.gamma = 1 # discount factor
        self.p = 0.4 # probability of home team winning
        self.num_states = 100 # the number of states available
        self.reward = [0 for _ in range(101)] # list for storing the reward values
        self.reward[100] = 1
        self.theta = 0.00000001 # small threshold value for comparing the difference
        self.value = [0 for _ in range(101)] # the value function for all states
        self.policy = [0 for _ in range(101)] # the amount of bet that gives the max reward

    def bellman_equation(self, num):
        """
        DOCSTRING
        """
        optimalvalue = 0
        for bet in range(0, min(num, 100 - num) + 1):
            win = num + bet
            loss = num - bet
            # calculate the average of possible states for an action
            sum = (
                self.p * (self.reward[win] + self.gamma * self.value[win]) + 
                (1 - self.p) * (self.reward[loss] + self.gamma * self.value[loss]))
            # choose the action that gives max reward and update the policy
            if sum > optimalvalue:
                optimalvalue = sum
                self.value[num] = sum
                self.policy[num] = bet

    def reinforcement_learning(self):
        """
        DOCSTRING
        """
        delta = 1
        while delta > self.theta:
            delta = 0
            for i in range(1, self.num_states):
                oldvalue = self.value[i]
                self.bellman_equation(i)
                diff = abs(oldvalue-self.value[i])
                delta = max(delta, diff)
        print(self.value)

if __name__ == '__main__':
    sports_betting = SportsBetting()
    sports_betting.reinforcement_learning()
