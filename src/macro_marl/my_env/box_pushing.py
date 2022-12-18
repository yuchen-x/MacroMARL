#!/usr/bin/python

import gym
import numpy as np
import IPython

from gym import spaces
from .box_pushing_core import Agent, Box 

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ACTIONS = ["Move_Forward", "Turn_L", "Turn_R", "Stay"]

class BoxPushing(gym.Env):
    
    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, grid_dim=(4,4), 
                 n_agent=2,
                 terminate_step=100, 
                 random_init=False, 
                 small_box_only=False, 
                 big_box_only=False,
                 terminal_reward_only=False, 
                 big_box_reward=100, 
                 small_box_reward=10, 
                 penalty=-5, 
                 *args, 
                 **kwargs):

        self.n_agent = n_agent
        assert n_agent <= 3, "TOO MANY AGENTS"
        
        #"move forward, turn left, turn right, stay"
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(5)

        self.xlen, self.ylen = grid_dim

        self.random_init = random_init
        self.small_box_only = small_box_only
        self.big_box_only = big_box_only
        assert not (self.big_box_only and self.small_box_only), "No box in the envrionment!"
        self.terminal_reward_only = terminal_reward_only
        self.big_box_reward = big_box_reward
        self.small_box_reward = small_box_reward
        self.penalty = penalty

        self.createAgents()
        self.createBoxes()
        
        self.terminate_step = terminate_step
        self.pushing_big_box = False

        self.viewer = None

        self.single_small_box = 0.0
        self.both_small_box = 0.0
        self.big_box = 0.0

    @property
    def state_size(self):
        return len(self.get_state())
    
    @property
    def obs_size(self):
        return [self.observation_space.n] * self.n_agent
    
    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]
    
    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    @property
    def action_spaces(self):
        return [self.action_space] * self.n_agent

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def get_state(self):
        positions = []
        for ag in self.agents:
            positions.append(ag.xcoord/self.xlen)
            positions.append(ag.ycoord/self.ylen)
            ori = np.eye(4)
            ori = ori[ag.ori]
            positions += ori.tolist()
        for bx in self.boxes:
            positions.append(bx.xcoord/self.xlen)
            positions.append(bx.ycoord/self.ylen)
        return np.array(positions)

    def createAgents(self):
        if self.random_init:
            init_ori = np.random.randint(4,size=2)
            init_xs = np.random.randint(self.xlen,size=2) + 0.5
            init_ys = np.random.randint(int(self.ylen/2-1),size=2) + 0.5
            A0 = Agent(0, init_xs[0], init_ys[0], init_ori[0], (self.xlen, self.ylen), self.penalty)
            A1 = Agent(1, init_xs[1], init_ys[1], init_ori[1], (self.xlen, self.ylen), self.penalty)
        else:
            if self.ylen >= 8.0:
                A0 = Agent(0, 1.5, 1.5, 1, (self.xlen, self.ylen), self.penalty)
                A1 = Agent(1, self.ylen-1.5, 1.5, 3, (self.xlen, self.ylen), self.penalty)
            elif self.ylen == 6.0:
                A0 = Agent(0, 0.5, 1.5, 1, (self.xlen, self.ylen), self.penalty)
                A1 = Agent(1, 5.5, 1.5, 3, (self.xlen, self.ylen), self.penalty)
            else:
                A0 = Agent(0, 0.5, 0.5, 1, (self.xlen, self.ylen), self.penalty)
                A1 = Agent(1, 3.5, 0.5, 3, (self.xlen, self.ylen), self.penalty)

        self.agents = [A0, A1]
        if self.n_agent == 3:
            if self.ylen == 4:
                A = Agent(3, self.xlen/2.0-0.5, 0.5, 0, (self.xlen, self.ylen), self.penalty)
            else:
                A = Agent(3, self.xlen/2.0-0.5, 1.5, 0, (self.xlen, self.ylen), self.penalty)
            self.agents.append(A)
    
    def createBoxes(self):
        if self.ylen >= 8.0:
            if not self.big_box_only:
                SB_0 = Box(0, 1.5, self.ylen/2+0.5, 1.0, 1.0) 
                SB_1 = Box(1, self.ylen-1.5, self.ylen/2+0.5, 1.0, 1.0) 
            if not self.small_box_only:
                BB_2 = Box(2, self.ylen/2.0, self.ylen/2+0.5, 1.0, 2.0) 
        elif self.ylen == 6.0:
            if not self.big_box_only:
                SB_0 = Box(0, 0.5, self.ylen/2+0.5, 1.0, 1.0) 
                SB_1 = Box(1, 5.5, self.ylen/2+0.5, 1.0, 1.0) 
            if not self.small_box_only:
                BB_2 = Box(2, 3.0, self.ylen/2+0.5, 1.0, 2.0) 
        else:
            if not self.big_box_only:
                SB_0 = Box(0, 0.5, self.ylen/2+0.5, 1.0, 1.0) 
                SB_1 = Box(1, 3.5, self.ylen/2+0.5, 1.0, 1.0) 
            if not self.small_box_only:
                BB_2 = Box(2, 2.0, self.ylen/2+0.5, 1.0, 2.0) 

        if self.small_box_only:
            self.boxes = [SB_0, SB_1]
        elif self.big_box_only:
            self.boxes = [BB_2]
        else:
            self.boxes = [SB_0, SB_1, BB_2]
    
    def reset(self, debug=False):
        self.createAgents()
        self.createBoxes()
        self.t = 0
        self.count_step = 0
        self.pushing_big_box = False

        if debug:
            self.render()

        return self._getobs()

    def step(self, actions, debug=False):

        if self.terminal_reward_only:
            rewards = 0.0
        else:
            rewards = -0.1
        terminate = 0

        cur_actions = actions
        cur_actions_done = [1] * self.n_agent
        self.pushing_big_box = False

        self.count_step += 1

        if not self.small_box_only and (actions[0] == 0) and (actions[1] == 0) and \
                self.agents[0].ori == 0 and self.agents[1].ori == 0 and \
                ((self.agents[0].xcoord == self.boxes[-1].xcoord-0.5 and \
                  self.agents[1].xcoord == self.boxes[-1].xcoord+0.5 and \
                  self.agents[0].ycoord == self.boxes[-1].ycoord-1.0 and \
                  self.agents[1].ycoord == self.boxes[-1].ycoord-1.0) or \
                 (self.agents[1].xcoord == self.boxes[-1].xcoord-0.5 and \
                  self.agents[0].xcoord == self.boxes[-1].xcoord+0.5 and \
                  self.agents[1].ycoord == self.boxes[-1].ycoord-1.0 and \
                  self.agents[0].ycoord == self.boxes[-1].ycoord-1.0)):
                    self.agents[0].pushing_big_box = True
                    self.agents[1].pushing_big_box = True
                    self.pushing_big_box = True
        if self.n_agent == 3:
            if not self.small_box_only and (actions[0] == 0) and (actions[2] == 0) and \
                    self.agents[0].ori == 0 and self.agents[2].ori == 0 and \
                    ((self.agents[0].xcoord == self.boxes[-1].xcoord-0.5 and \
                    self.agents[2].xcoord == self.boxes[-1].xcoord+0.5 and \
                    self.agents[0].ycoord == self.boxes[-1].ycoord-1.0 and \
                    self.agents[2].ycoord == self.boxes[-1].ycoord-1.0) or \
                    (self.agents[2].xcoord == self.boxes[-1].xcoord-0.5 and \
                    self.agents[0].xcoord == self.boxes[-1].xcoord+0.5 and \
                    self.agents[2].ycoord == self.boxes[-1].ycoord-1.0 and \
                    self.agents[0].ycoord == self.boxes[-1].ycoord-1.0)):
                        self.agents[0].pushing_big_box = True
                        self.agents[2].pushing_big_box = True
                        self.pushing_big_box = True

            if not self.small_box_only and (actions[1] == 0) and (actions[2] == 0) and \
                    self.agents[1].ori == 0 and self.agents[2].ori == 0 and \
                    ((self.agents[1].xcoord == self.boxes[-1].xcoord-0.5 and \
                    self.agents[2].xcoord == self.boxes[-1].xcoord+0.5 and \
                    self.agents[1].ycoord == self.boxes[-1].ycoord-1.0 and \
                    self.agents[2].ycoord == self.boxes[-1].ycoord-1.0) or \
                    (self.agents[2].xcoord == self.boxes[-1].xcoord-0.5 and \
                    self.agents[1].xcoord == self.boxes[-1].xcoord+0.5 and \
                    self.agents[2].ycoord == self.boxes[-1].ycoord-1.0 and \
                    self.agents[1].ycoord == self.boxes[-1].ycoord-1.0)):
                        self.agents[1].pushing_big_box = True
                        self.agents[2].pushing_big_box = True
                        self.pushing_big_box = True

        for idx, agent in enumerate(self.agents):
            if agent.pushing_big_box:
                agent.cur_action = 0
                agent.ycoord += 1.0
            else:
                reward = agent.step(actions[idx], self.boxes)
                if not self.terminal_reward_only:
                    rewards += reward

        if self.pushing_big_box:
            self.boxes[-1].ycoord += 1.0

        # check whether any box is pushed to the goal area
        reward = 0.0
        small_box = 0.0

        for idx, box in enumerate(self.boxes):
            if box.ycoord == self.ylen - 0.5:
                terminate = 1
                reward = reward + self.small_box_reward if idx < 2 else reward + self.big_box_reward
                if idx == 2:
                    self.big_box += 1.0
                else:
                    small_box += 1.0

        if small_box == 1.0:
            self.single_small_box += 1.0
        elif small_box == 2.0:
            self.both_small_box += 1.0

        rewards += reward

        if debug:
            self.render()
            print("Actions list:")
            for i in range(self.n_agent):
                print(" ")
                print("Agent_", i, " \t action \t\t{}".format(ACTIONS[self.agents[i].cur_action]))

        observations = self._getobs(debug)

        return observations, [rewards]*self.n_agent, bool(terminate), {"cur_mac": cur_actions, "mac_done": cur_actions_done}

    def _getobs(self, debug=False):

        if debug:
            print("")
            print("Observations list:")

        observations = []
        for idx, agent in enumerate (self.agents):

            obs = np.zeros(self.observation_space.n)

            # assume empty front
            obs[2] = 1.0

            if not self.big_box_only:
                # observe small box
                for box in self.boxes[0:2]:
                    if box.xcoord == agent.xcoord + DIRECTION[agent.ori][0] and \
                            box.ycoord == agent.ycoord + DIRECTION[agent.ori][1]:
                                obs[0] = 1.0
                                obs[2] = 0.0
            if not self.small_box_only:
                # observe large box
                if (self.boxes[-1].xcoord+0.5 == agent.xcoord + DIRECTION[agent.ori][0] or \
                    self.boxes[-1].xcoord-0.5 == agent.xcoord + DIRECTION[agent.ori][0]) and \
                    self.boxes[-1].ycoord  == agent.ycoord + DIRECTION[agent.ori][1]:
                            obs[1] = 1.0
                            obs[2] = 0.0
           
            # observe wall
            if agent.xcoord + DIRECTION[agent.ori][0] > self.xlen or \
                    agent.xcoord + DIRECTION[agent.ori][0] < 0.0 or \
                    agent.ycoord + DIRECTION[agent.ori][1] > self.ylen or \
                    agent.ycoord + DIRECTION[agent.ori][1] < 0.0:
                        obs[3] = 1.0
                        obs[2] = 0.0
            
            # observe agent
            teamates = list(range(self.n_agent))
            teamates.remove(idx)
            for teamate_idx in teamates:
                if (agent.xcoord + DIRECTION[agent.ori][0] == self.agents[teamate_idx].xcoord) and \
                        (agent.ycoord + DIRECTION[agent.ori][1] == self.agents[teamate_idx].ycoord):
                    obs[4] = 1.0
                    obs[2] = 0.0
                    break

            if debug:
                    print("Agent_" + str(idx) + " \t small_box  \t\t{}".format(obs[0]))
                    print("          " + " \t large_box \t\t{}".format(obs[1]))
                    print("          " + " \t empty \t\t\t{}".format(obs[2]))
                    print("          " + " \t wall \t\t\t{}".format(obs[3]))
                    print("          " + " \t teammate \t\t{}".format(obs[4]))
                    print("")

            observations.append(obs)

        self.old_observations = observations

        return observations

    def get_state(self):
        positions = []
        for ag in self.agents:
            positions.append(ag.xcoord/self.xlen)
            positions.append(ag.ycoord/self.ylen)
            ori = np.eye(4)
            ori = ori[ag.ori]
            positions += ori.tolist()
        for bx in self.boxes:
            positions.append(bx.xcoord/self.xlen)
            positions.append(bx.ycoord/self.ylen)
        return np.array(positions)

    def get_env_info(self):
        return {'state_shape': len(self.get_state()),
                'obs_shape': self.obs_size[0],
                'n_actions': self.n_action[0],
                'n_agents': self.n_agent,
                'episode_limit': self.terminate_step}

    def get_avail_actions(self):
        return [[1]*n for n in self.n_action]

    def render(self, mode='human'):
        
        screen_width = 8*100
        screen_height = 8*100

        scale = 8 / self.ylen

        agent_size = 30.0
        agent_in_size = 25.0
        agent_clrs = [((0.15,0.65,0.15), (0.0,0.8,0.4)), ((0.15,0.15,0.65), (0.0, 0.4,0.8)), ((0.2,0.0,0.4), (0.5,0.0,1.0)), ((0.12,0.12,0.12), (0.5,0.5,0.5)), ((0,0.4,0.4), (0.0,1.0,1.0)), ((0.4,0.2,0.0), (1.0,0.5,0.0))]

        small_box_size = 85.0
        small_box_clrs = [(0.43,0.28,0.02), (0.67,0.43,0.02)]
        small_box_in_size = 75.0

        big_box_l = 185.0
        big_box_in_l = 175.0
        big_box_w = 85.0
        big_box_in_w = 75.0
        big_box_clrs = [(0.43,0.28,0.02), (0.67,0.43,0.02)]
        
        if self.viewer is None:
            from macro_marl.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #-------------------draw line-----------------
            for l in range(1, self.ylen):
                line = rendering.Line((0.0, l*100*scale), (screen_width, l*100*scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.ylen):
                line = rendering.Line((l*100*scale, 0.0), (l*100*scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (0.0, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_width), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #-------------------draw goal
            goal = rendering.FilledPolygon([(-(screen_width-8)/2.0,(-50+2)*scale), (-(screen_width-8)/2.0,(50-2)*scale), ((screen_width-8)/2.0,(50-2)*scale), ((screen_width-8)/2.0,-(50-2)*scale)])
            goal.set_color(1.0,1.0,0.0)
            goal_trans = rendering.Transform(translation=(screen_width/2.0,(self.ylen-0.5)*100*scale))
            goal.add_attr(goal_trans)
            self.viewer.add_geom(goal)

            #-------------------draw small box
            if not self.big_box_only:
                small_box_0 = rendering.FilledPolygon([(-small_box_size/2.0*scale,-small_box_size/2.0*scale), (-small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,-small_box_size/2.0*scale)])
                small_box_0.set_color(*small_box_clrs[0])
                self.small_box_0_trans = rendering.Transform(translation=(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale))
                small_box_0.add_attr(self.small_box_0_trans)
                self.viewer.add_geom(small_box_0)

                small_box_0_in = rendering.FilledPolygon([(-small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale), (-small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale)])
                small_box_0_in.set_color(*small_box_clrs[1])
                self.small_box_0_in_trans = rendering.Transform(translation=(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale))
                small_box_0_in.add_attr(self.small_box_0_in_trans)
                self.viewer.add_geom(small_box_0_in)
                
                small_box_1 = rendering.FilledPolygon([(-small_box_size/2.0*scale,-small_box_size/2.0*scale), (-small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,-small_box_size/2.0*scale)])
                small_box_1.set_color(*small_box_clrs[0])
                self.small_box_1_trans = rendering.Transform(translation=(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale))
                small_box_1.add_attr(self.small_box_1_trans)
                self.viewer.add_geom(small_box_1)

                small_box_1_in = rendering.FilledPolygon([(-small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale), (-small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale)])
                small_box_1_in.set_color(*small_box_clrs[1])
                self.small_box_1_in_trans = rendering.Transform(translation=(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale))
                small_box_1_in.add_attr(self.small_box_1_in_trans)
                self.viewer.add_geom(small_box_1_in)

            # -------------------draw big box
            if not self.small_box_only:
                big_box_2 = rendering.FilledPolygon([(-big_box_l/2.0*scale,-big_box_w/2.0*scale), (-big_box_l/2.0*scale,big_box_w/2.0*scale), (big_box_l/2.0*scale,big_box_w/2.0*scale), (big_box_l/2.0*scale,-big_box_w/2.0*scale)])
                big_box_2.set_color(*big_box_clrs[0])
                self.big_box_2_trans = rendering.Transform(translation=(self.boxes[-1].xcoord*100*scale, self.boxes[-1].ycoord*100*scale))
                big_box_2.add_attr(self.big_box_2_trans)
                self.viewer.add_geom(big_box_2)

                big_box_2_in = rendering.FilledPolygon([(-big_box_in_l/2.0*scale,-big_box_in_w/2.0*scale), (-big_box_in_l/2.0*scale,big_box_in_w/2.0*scale), (big_box_in_l/2.0*scale,big_box_in_w/2.0*scale), (big_box_in_l/2.0*scale,-big_box_in_w/2.0*scale)])
                big_box_2_in.set_color(*big_box_clrs[1])
                self.big_box_2_in_trans = rendering.Transform(translation=(self.boxes[-1].xcoord*100*scale, self.boxes[-1].ycoord*100*scale))
                big_box_2_in.add_attr(self.big_box_2_in_trans)
                self.viewer.add_geom(big_box_2_in)

            #-------------------draw agent
            self.agent_trans = []
            for ag in self.agents:
                agent = rendering.make_circle(radius=agent_size*scale)
                agent.set_color(*agent_clrs[ag.idx][0])
                self.agent_trans.append(rendering.Transform(translation=(ag.xcoord*100*scale, ag.ycoord*100*scale)))
                agent.add_attr(self.agent_trans[-1])
                self.viewer.add_geom(agent)

            self.agent_in_trans = []
            for ag in self.agents:
                agent_in = rendering.make_circle(radius=agent_in_size*scale)
                agent_in.set_color(*agent_clrs[ag.idx][1])
                self.agent_in_trans.append(rendering.Transform(translation=(ag.xcoord*100*scale, ag.ycoord*100*scale)))
                agent_in.add_attr(self.agent_in_trans[-1])
                self.viewer.add_geom(agent_in)
 
            #-------------------draw agent sensor
            sensor_size = 20.0
            sensor_in_size = 14.0
            sensor_clrs = ((0.65,0.15,0.15), (1.0, 0.2,0.2))

            self.sensor_trans = []
            for idx in range(self.n_agent):
                sensor = rendering.FilledPolygon([(-sensor_size/2.0*scale,-sensor_size/2.0*scale), (-sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,-sensor_size/2.0*scale)])
                sensor.set_color(*sensor_clrs[0])
                self.sensor_trans.append(rendering.Transform(translation=(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                                                       self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)))
                sensor.add_attr(self.sensor_trans[-1])
                self.viewer.add_geom(sensor)

            self.sensor_in_trans = []
            for idx in range(self.n_agent):
                sensor_in = rendering.FilledPolygon([(-sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale), (-sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale)])
                sensor_in.set_color(*sensor_clrs[1])
                self.sensor_in_trans.append(rendering.Transform(translation=(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                                                            self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)))
                sensor_in.add_attr(self.sensor_in_trans[-1])
                self.viewer.add_geom(sensor_in)
 
        if not self.big_box_only:
            self.small_box_0_trans.set_translation(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale)
            self.small_box_0_in_trans.set_translation(self.boxes[0].xcoord*100*scale, self.boxes[0].ycoord*100*scale)
            self.small_box_1_trans.set_translation(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale)
            self.small_box_1_in_trans.set_translation(self.boxes[1].xcoord*100*scale, self.boxes[1].ycoord*100*scale)
        if not self.small_box_only:
            self.big_box_2_trans.set_translation(self.boxes[-1].xcoord*100*scale, self.boxes[-1].ycoord*100*scale)
            self.big_box_2_in_trans.set_translation(self.boxes[-1].xcoord*100*scale, self.boxes[-1].ycoord*100*scale)
        
        for idx, trans in enumerate(self.agent_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale, self.agents[idx].ycoord*100*scale)
        for idx, trans in enumerate(self.agent_in_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale, self.agents[idx].ycoord*100*scale)

        for idx, trans in enumerate(self.sensor_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)
            trans.set_rotation(0.0)
        for idx, trans in enumerate(self.sensor_in_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)
            trans.set_rotation(0.0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
