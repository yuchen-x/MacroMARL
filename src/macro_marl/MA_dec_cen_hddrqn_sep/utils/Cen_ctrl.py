class Cen_Controller:

    def __init__(self):
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.loss = None
