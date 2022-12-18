class Agent:

    """A class of an agent"""

    def __init__(self):
        self.idx = None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.loss = None
