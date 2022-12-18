#!/usr/bin/python
import gym
import numpy as np
import IPython

from .core_double_room import AgentTurtlebot_v4, AgentTurtlebot_v7, AgentFetch_v4, AgentHuman, BeliefWayPoint, MacroAction
from gym import spaces

class ObjSearchDelivery(gym.Env):

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, 
                 n_objs=3, 
                 terminate_step=150, 
                 human_speed_per_step=[[18,18,18,18], [18,18,18,18]], 
                 TB_move_speed=0.6, 
                 TB_move_noisy=0.0, 
                 TB_move_cost=0.0,
                 fetch_pass_obj_tc=4, 
                 fetch_look_for_obj_tc=6, 
                 fetch_manip_noisy=0.0, 
                 fetch_drop_obj_penalty=-10.0, 
                 delay_delivery_penalty=0.0, 
                 obs_human_wait=False, 
                 render=False,
                 *args, 
                 **kwargs):

        self.n_agent = 3

        #-----------------basic settings for this domain
        self.n_objs = n_objs                             # define the number of different objects needs for each human to finish the whole task
                                                         # remeber change the corresponding argument when change this parameter
        self.n_each_obj = 2                              # total amount of each obj in the env
        self.n_steps_human_task = self.n_objs + 1        # define the number of steps for each human finishing the task 

        #-----------------def belief waypoints
        BWP0 = BeliefWayPoint('WorkArea0', 0, 6.0, 3.0)
        BWP1 = BeliefWayPoint('WorkArea1', 1, 6.0, 2.0)
        BWP2 = BeliefWayPoint('ToolRoomWait', 2, 3.0, 2.5)
        BWP3_T0 = BeliefWayPoint('ToolRoomTable', 3, 1.2, 3.7)
        BWP3_T1 = BeliefWayPoint('ToolRoomTable', 3, 1.2, 1.3)

        self.BWPs = [BWP0, BWP1, BWP2, BWP3_T0, BWP3_T1]
        self.BWPs_T0 = [BWP0, BWP1, BWP2, BWP3_T0]
        self.BWPs_T1 = [BWP0, BWP1, BWP2, BWP3_T1]
        
        self.viewer = None
        self.terminate_step = terminate_step

        self.TB_move_speed = TB_move_speed
        self.TB_move_noisy = TB_move_noisy
        self.TB_move_cost = TB_move_cost
        self.fetch_pass_obj_tc = fetch_pass_obj_tc
        self.fetch_look_for_obj_tc = fetch_look_for_obj_tc
        self.fetch_manip_noisy = fetch_manip_noisy
        self.fetch_drop_obj_penalty = fetch_drop_obj_penalty
        self.human_speed = human_speed_per_step

        self.finish_serve_humans = []
        self.obs_human_wait = obs_human_wait

        self.delay_delivery_penalty = delay_delivery_penalty
        self.rendering = render

    @property
    def state_size(self):
        return len(self.get_state())

    @property
    def obs_size(self):
        return [self.observation_space_T.n] *2 + [self.observation_space_F.n]

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    @property
    def action_spaces(self):
        return [self.action_space_T] * 2 + [self.action_space_F]

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agent)]

    def get_avail_agent_actions(self, nth):
        return [1] * self.action_spaces[nth].n

    def get_state(self):
        raise NotImplementedError

    def create_turtlebot_actions(self):
        raise NotImplementedError

    def create_fetch_actions(self):
        raise NotImplementedError

    def createAgents(self):
        raise NotImplementedError

    def createHumans(self):

        #-----------------initialize Three Humans
        Human0 = AgentHuman(0, self.n_steps_human_task, self.human_speed[0], list(range(self.n_objs)))
        #Human1 = AgentHuman(1, self.n_steps_human_task, [self.human_speed[1]]*self.n_steps_human_task, list(range(self.n_objs)))
        Human1 = AgentHuman(1, self.n_steps_human_task, self.human_speed[1], list(range(self.n_objs)))

        self.humans = [Human0, Human1]
        self.n_human = len(self.humans)
        self.n_human_finished = []   # recording the number of human who has finished his own task
        
    def step(self, actions):
        raise NotImplementedError

    def reset(self):
        
        # reset the agents in this env
        self.createAgents()

        # reset the humans in this env
        for human in self.humans:
            human.reset()
        self.n_human_finished = []
        self.finish_serve_humans = []
        
        self.t = 0   # indicates the beginning of one episode, check _getobs()
        self.count_step = 0

        if self.rendering:
            self.render()

        return self._getobs()

    def _getobs(self):
        raise NotImplementedError

    def render(self, mode='human'):

        screen_width = 700
        screen_height = 500

        if self.viewer is None:
            from macro_marl.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            line = rendering.Line((0.0, 0.0), (0.0, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_height), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #--------------------------------draw rooms' boundaries

            for i in range(0,100,2):
                line_tool_room = rendering.Line((350, i*5), (350, (i+1)*5))
                line_tool_room.set_color(0,0,0)
                line_tool_room.linewidth.stroke = 2
                self.viewer.add_geom(line_tool_room)

            for i in range(0,80,2):
                line_WA = rendering.Line((500+i*5, 300), (500+(i+1)*5, 300))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,80,2):
                line_WA = rendering.Line((500, 300+i*5), (500, 300+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
                
            for i in range(0,80,2):
                line_WA = rendering.Line((500+i*5, 200), (500+(i+1)*5, 200))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,40,2):
                line_WA = rendering.Line((500, 0+i*5), (500, 0+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
            
            #---------------------------draw BW0
            for i in range(len(self.BWPs)):
                BWP = rendering.make_circle(radius=6)
                BWP.set_color(178.0/255.0, 34.0/255.0, 34.0/255.0)
                BWPtrans = rendering.Transform(translation=(self.BWPs[i].xcoord*100, self.BWPs[i].ycoord*100))
                BWP.add_attr(BWPtrans)
                self.viewer.add_geom(BWP)

            #-------------------------------draw table
            tablewidth = 60.0
            tableheight = 125.0
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.43,0.28,0.02)
            tabletrans = rendering.Transform(translation=(175, 250))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            tablewidth = 54.0
            tableheight = 119.0
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.67,0.43,0.02)
            tabletrans = rendering.Transform(translation=(175, 250))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            #-----------------------------draw Fetch
            fetch = rendering.make_circle(radius=26)
            fetch.set_color(*(0.0,0.0,0.0))
            self.fetchtrans = rendering.Transform(translation=(self.agents[2].xcoord*100, self.agents[2].ycoord*100))
            fetch.add_attr(self.fetchtrans)
            self.viewer.add_geom(fetch)

            #-----------------------------draw Fetch
            fetch_c = rendering.make_circle(radius=23)
            fetch_c.set_color(*(0.5, 0.5,0.5))
            self.fetchtrans_c = rendering.Transform(translation=(self.agents[2].xcoord*100, self.agents[2].ycoord*100))
            fetch_c.add_attr(self.fetchtrans_c)
            self.viewer.add_geom(fetch_c)

            #-----------------------------draw Fetch arms
            self.arm2 = rendering.FilledPolygon([(-5.0,-20.0,), (-5.0, 20.0), (5.0, 20.0), (5.0, -20.0)])
            self.arm2.set_color(0.0, 0.0, 0.0)
            self.arm2trans = rendering.Transform(translation=(self.agents[2].xcoord*10000+49, self.agents[2].ycoord*100), rotation = -90/180*np.pi)
            self.arm2.add_attr(self.arm2trans)
            self.viewer.add_geom(self.arm2)

            self.arm2_c = rendering.FilledPolygon([(-3.0,-18.0,), (-3.0, 18.0), (3.0, 18.0), (3.0, -18.0)])
            self.arm2_c.set_color(0.5, 0.5, 0.5)
            self.arm2trans_c = rendering.Transform(translation=(self.agents[2].xcoord*10000+48, self.agents[2].ycoord*100), rotation = -90/180*np.pi)
            self.arm2_c.add_attr(self.arm2trans_c)
            self.viewer.add_geom(self.arm2_c)

            self.arm1 = rendering.FilledPolygon([(-5.0,-38.0,), (-5.0, 38.0), (5.0, 38.0), (5.0, -38.0)])
            self.arm1.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 187), rotation = 15/180*np.pi)
            self.arm1.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1)

            self.arm1_c = rendering.FilledPolygon([(-3.0,-36.0,), (-3.0, 36.0), (3.0, 36.0), (3.0, -36.0)])
            self.arm1_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 187), rotation = 15/180*np.pi)
            self.arm1_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1_c)
            
            self.arm0 = rendering.FilledPolygon([(-5.0,-38.0,), (-5.0, 38.0), (5.0, 38.0), (5.0, -38.0)])
            self.arm0.set_color(1.0, 1.0, 1.0)
            arm0trans = rendering.Transform(translation=(108, 313), rotation = -15/180*np.pi)
            self.arm0.add_attr(arm0trans)
            self.viewer.add_geom(self.arm0)

            self.arm0_c = rendering.FilledPolygon([(-3.0,-36.0,), (-3.0, 36.0), (3.0, 36.0), (3.0, -36.0)])
            self.arm0_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 313), rotation = -15/180*np.pi)
            self.arm0_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm0_c)

            #----------------------------draw Turtlebot_1
            turtlebot_1 = rendering.make_circle(radius=17.0)
            turtlebot_1.set_color(*(0.15,0.65,0.15))
            self.turtlebot_1trans = rendering.Transform(translation=(self.agents[0].xcoord*100, self.agents[0].ycoord*100))
            turtlebot_1.add_attr(self.turtlebot_1trans)
            self.viewer.add_geom(turtlebot_1)

            turtlebot_1_c = rendering.make_circle(radius=14.0)
            turtlebot_1_c.set_color(*(0.0,0.8,0.4))
            self.turtlebot_1trans_c = rendering.Transform(translation=(self.agents[0].xcoord*100, self.agents[0].ycoord*100))
            turtlebot_1_c.add_attr(self.turtlebot_1trans_c)
            self.viewer.add_geom(turtlebot_1_c)
            
            #----------------------------draw Turtlebot_2
            turtlebot_2 = rendering.make_circle(radius=17.0)
            turtlebot_2.set_color(*(0.15,0.15,0.65))
            self.turtlebot_2trans = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100))
            turtlebot_2.add_attr(self.turtlebot_2trans)
            self.viewer.add_geom(turtlebot_2)

            turtlebot_2_c = rendering.make_circle(radius=14.0)
            turtlebot_2_c.set_color(*(0.0,0.4,0.8))
            self.turtlebot_2trans_c = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100))
            turtlebot_2_c.add_attr(self.turtlebot_2trans_c)
            self.viewer.add_geom(turtlebot_2_c)

            #----------------------------draw human_0's status
            self.human0_progress_bar = []
            total_steps = self.humans[0].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10,-10), (-10,10), (10,10), (10,-10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26,480))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human0_progress_bar.append(progress_bar)
            
            #----------------------------draw human_1's status
            self.human1_progress_bar = []
            total_steps = self.humans[1].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10,-10), (-10,10), (10,10), (10,-10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26,20))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human1_progress_bar.append(progress_bar)
            

        # draw each robot's status
        self.turtlebot_1trans.set_translation(self.agents[0].xcoord*100, self.agents[0].ycoord*100)
        self.turtlebot_1trans_c.set_translation(self.agents[0].xcoord*100, self.agents[0].ycoord*100)
        self.turtlebot_2trans.set_translation(self.agents[1].xcoord*100, self.agents[1].ycoord*100)
        self.turtlebot_2trans_c.set_translation(self.agents[1].xcoord*100, self.agents[1].ycoord*100)
        self.fetchtrans.set_translation(self.agents[2].xcoord*100, self.agents[2].ycoord*100)

        # # reset human's progress bar
        for idx, bar in enumerate(self.human0_progress_bar):
            bar.set_color(0.8,0.8,0.8)
        for idx, bar in enumerate(self.human1_progress_bar):
            bar.set_color(0.8,0.8,0.8)

        # draw each human's status
        if self.humans[0].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx < self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[0].cur_step:
                    bar.set_color(0.0, 1.0, 0.0)
                    break
        else:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx <= self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # draw each human's status
        if self.humans[1].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx < self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[1].cur_step:
                    bar.set_color(0.0, 1.0, 0.0)
                    break
        else:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx <= self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # reset fetch arm
        self.arm0.set_color(1.0, 1.0, 1.0)
        self.arm0_c.set_color(1.0, 1.0, 1.0)
        self.arm1.set_color(1.0, 1.0, 1.0)
        self.arm1_c.set_color(1.0, 1.0, 1.0)

        self.arm2trans_c.set_translation(self.agents[2].xcoord*10000+48, self.agents[2].ycoord*100)
        self.arm2trans.set_translation(self.agents[2].xcoord*10000+49, self.agents[2].ycoord*100)


        self.pass_objs = 0

        if self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 1 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    self.pass_objs += 1
                    self.arm0.set_color(0.0, 0.0, 0.0)
                    self.arm0_c.set_color(0.5,0.5,0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 2 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    self.pass_objs += 1
                    self.arm1.set_color(0.0, 0.0, 0.0)
                    self.arm1_c.set_color(0.5, 0.5, 0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx > 2 and \
                np.sum(self.agents[2].count_found_obj) < self.n_objs*self.n_each_obj:
                    self.arm2trans_c.set_translation(self.agents[2].xcoord*100+48, self.agents[2].ycoord*100)
                    self.arm2trans.set_translation(self.agents[2].xcoord*100+49, self.agents[2].ycoord*100)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class ObjSearchDelivery_v7(ObjSearchDelivery):

    """1) Not distinguish Look_for_obj to different robot.
       2) Turtlebot get tool and wait until fetch pass any obj to it or terminate it in 10s.
       3) Turtlebot observes human status but doesn't not have request tool action
      *4) Fetch observes which object on the waiting spots instead of how many object on it
       5) Fixed Penalty"""

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, *args, **kwargs):

        super(ObjSearchDelivery_v7, self).__init__(*args, **kwargs)

        self.createHumans()
        self.create_turtlebot_actions()
        self.create_fetch_actions()
        self.createAgents()

    def create_turtlebot_actions(self):

        self.action_space_T = spaces.Discrete(4)

        if not self.obs_human_wait:
            # Descrete location areas: 3 
            # human's working step
            # which object in the basket: n_objs
            # which turtlebot's object is ready: 2
            self.observation_space_T = spaces.MultiBinary(5+self.n_steps_human_task*self.n_human+self.n_objs+2)
        else:
            self.observation_space_T = spaces.MultiBinary(5+self.n_steps_human_task*self.n_human+self.n_objs+2+self.n_human)
        #-----------------def single step macro-actions for Turtlebot 
        T_MA0 = MacroAction('Go_WA0', 0, expected_t_cost = None, ma_bwpterm = 0)
        T_MA1 = MacroAction('Go_WA1', 1, expected_t_cost = None, ma_bwpterm = 1)
        T_MA2 = MacroAction('Go_Tool_Room', 2, expected_t_cost = None, ma_bwpterm = 2)
        T_MA3 = MacroAction('Get_Tool', 3, expected_t_cost = None, ma_bwpterm = 3)

        self.T_MAs = [T_MA0, T_MA1, T_MA2, T_MA3]

    def create_fetch_actions(self):
        self.action_space_F = spaces.Discrete(3+self.n_objs)
        if not self.obs_human_wait:
            # which object is on the waiting spot
            # which turtlebot is beside the table: 2
            self.observation_space_F = spaces.MultiBinary(self.n_objs+2) 
        else:
            self.observation_space_F = spaces.MultiBinary(self.n_objs+2+self.n_human)
           
        #-----------------def single step macro-actions for Fetch Robot
        F_MA0 = MacroAction('Wait_Request', 0, expected_t_cost = 1)
        F_MA1 = MacroAction('Pass_Obj_T0', 1, expected_t_cost = self.fetch_pass_obj_tc)
        F_MA2 = MacroAction('Pass_Obj_T1', 2, expected_t_cost = self.fetch_pass_obj_tc)
        self.F_MAs = [F_MA0, F_MA1, F_MA2]
        for i in range(self.n_objs):
            F_MA = MacroAction('Look_For_obj'+str(i), i+3, expected_t_cost=self.fetch_look_for_obj_tc)
            self.F_MAs.append(F_MA)

    def createAgents(self):

        #-----------------initialize Two Turtlebot agents
        Turtlebot1 = AgentTurtlebot_v7(0, 3.0, 3.5, self.BWPs_T0, self.T_MAs, self.n_objs, speed=self.TB_move_speed, move_noisy=self.TB_move_noisy, 
                                       move_cost=self.TB_move_cost, get_tool_wait=self.fetch_pass_obj_tc+self.fetch_look_for_obj_tc, delay_delivery_penalty=self.delay_delivery_penalty)
        Turtlebot2 = AgentTurtlebot_v7(1, 3.0, 1.5, self.BWPs_T1, self.T_MAs, self.n_objs, speed=self.TB_move_speed, move_noisy=self.TB_move_noisy, 
                                       move_cost=self.TB_move_cost, get_tool_wait=self.fetch_pass_obj_tc+self.fetch_look_for_obj_tc, delay_delivery_penalty=self.delay_delivery_penalty)

        #-----------------initialize One Fetch Robot agent
        Fetch_robot = AgentFetch_v4(2, 0.9, 2.5, self.F_MAs, self.n_objs, self.n_each_obj, self.fetch_manip_noisy, self.fetch_drop_obj_penalty)

        self.agents = [Turtlebot1, Turtlebot2, Fetch_robot]

    def createHumans(self):

        #-----------------initialize Two Humans
        Human0 = AgentHuman(0, self.n_steps_human_task, self.human_speed[0], list(range(self.n_objs)))
        Human1 = AgentHuman(1, self.n_steps_human_task, self.human_speed[1], list(range(self.n_objs)))
        if sum(self.human_speed[0]) == sum(self.human_speed[1]):
            self.human_bar_colors = [(0.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        else:
            self.human_bar_colors = [(1.0, 0.5, 0.0), (0.0, 1.0, 0.0)]

        self.humans = [Human0, Human1]
        self.n_human = len(self.humans)
        self.n_human_finished = []   # recording the number of human who has finished his own task

    def step(self, actions):

        """
        Parameters
        ----------
        actions : int | List[..]
           The discrete macro-action index for one or more agents. 

        Returns
        -------
        observations : ndarry | List[..]
            A list of  each agent's macor-observation.
        rewards : float | List[..]
            A global shared reward for each agent.
        done : bool
            Whether the current episode is over or not.
        info: dict{}
            "mac_done": binary(1/0) | List[..]
                whether the current macro_action is done or not.
            "cur_mac": int | list[..]
                "The current macro-action's indices"
        """

        rewards = -1.0
        terminate = 0
        cur_actions= []
        cur_actions_done= []

        self.count_step += 1

        # Each Turtlebot executes one step
        for idx, turtlebot in enumerate(self.agents[0:2]):
            # when the previous macro-action has not been finished, and return the previous action id
            if not turtlebot.cur_action_done:
                reward = turtlebot.step(turtlebot.cur_action.idx, self.humans)
                cur_actions.append(turtlebot.cur_action.idx)
            else:
                reward = turtlebot.step(actions[idx], self.humans)
                cur_actions.append(actions[idx])
            rewards += reward

        # Fetch executes one step
        if not self.agents[2].cur_action_done:
            reward = self.agents[2].step(self.agents[2].cur_action.idx, self.agents)
            cur_actions.append(self.agents[2].cur_action.idx)
        else:
            reward = self.agents[2].step(actions[2], self.agents)
            cur_actions.append(actions[2])
        rewards += reward

        # collect the info about the cur_actions and if they are finished
        for idx, agent in enumerate(self.agents):
            cur_actions_done.append(1 if agent.cur_action_done else 0)

        # episode terminates condition
        for idx, human in enumerate(self.humans):
            if idx not in self.finish_serve_humans and \
               (human.cur_step == self.n_steps_human_task-2) and \
               human.next_requested_obj_obtained:
                self.finish_serve_humans.append(idx)

        # each human executes one step
        for idx, human in enumerate(self.humans):
            if idx in self.n_human_finished:
                continue
            human.cur_step_time_left -= 1.0

            if human.cur_step_time_left <= 0.0 and human.next_requested_obj_obtained:
                human.step()
            
            if human.whole_task_finished:
                self.n_human_finished.append(idx)

        if self.rendering:
            self.render()
            print(" ")
            print("Actions list:")
            print("Turtlebot0 \t action \t\t{}".format(self.agents[0].cur_action.name))
            print("           \t action_t_left \t\t{}".format(self.agents[0].cur_action_time_left))
            print("           \t action_done \t\t{}".format(self.agents[0].cur_action_done))
            print("Turtlebot1 \t action \t\t{}".format(self.agents[1].cur_action.name))
            print("           \t action_t_left \t\t{}".format(self.agents[1].cur_action_time_left))
            print("           \t action_done \t\t{}".format(self.agents[1].cur_action_done))
            print("Fetchrobot \t action \t\t{}".format(self.agents[2].cur_action.name))
            print("           \t action_t_left \t\t{}".format(self.agents[2].cur_action_time_left))
            print("           \t action_done \t\t{}".format(self.agents[2].cur_action_done))
            print("           \t is_serving \t\t{}".format(self.agents[2].serving))
            print("           \t serving_failed \t{}".format(self.agents[2].serving_failed))

        observations = self._getobs()

        # reset Turtlebot request.
        self.agents[0].request_objs = np.zeros(self.n_objs)
        self.agents[1].request_objs = np.zeros(self.n_objs)

        if self.rendering:
            print("")
            print("Humans status:")
            for idx, human in enumerate(self.humans):
                print("Human" + str(idx) + " \t\t cur_step  \t\t\t{}".format(human.cur_step))
                print("      " + " \t\t cur_step_t_left  \t\t{}".format(human.cur_step_time_left))
                print("      " + " \t\t next_request_obj  \t\t{}".format(human.next_request_obj_idx))
                print("      " + " \t\t requested_obj_obtain  \t\t{}".format(human.next_requested_obj_obtained))
                print("      " + " \t\t whole_task_finished  \t\t{}".format(human.whole_task_finished))
                print(" ")

        return observations, [rewards]*self.n_agent, len(self.finish_serve_humans) == 2, {'cur_mac': cur_actions, 'mac_done': cur_actions_done}

    def _getobs(self):

        #--------------------get observations at the beginning of each episode
        if self.t == 0:
            # get initial observation for turtlebot0
            T_obs_0 = np.zeros(self.observation_space_T.n)
            T_obs_0[len(self.BWPs_T0)] = 1.0

            # get initial observation for turtlebot1
            T_obs_1 = np.zeros(self.observation_space_T.n)
            T_obs_1[len(self.BWPs_T1)] = 1.0

            # get initial observaion for fetch robot
            F_obs = np.zeros(self.observation_space_F.n) 

            observations = [T_obs_0, T_obs_1, F_obs]
            self.t = 1
            self.old_observations = observations

            return observations

        #---------------------get observations for the two turtlebots
        if self.rendering:
            print("")
            print("observations list:")

        observations = []
        for idx, agent in enumerate(self.agents[0:2]):

            # won't get new obs until current macro-action finished
            if not agent.cur_action_done:
                observations.append(self.old_observations[idx])
                if self.rendering:
                    print("turtlebot" + str(idx) + " \t loc  \t\t\t{}".format(self.old_observations[idx][0:(len(self.BWPs_T0)+1)]))
                    print("          " + " \t hm0_cur_step \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1):
                                                                          (len(self.BWPs_T0)+1)+self.n_steps_human_task]))
                    print("          " + " \t hm1_cur_step \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1)+self.n_steps_human_task:
                                                                          (len(self.BWPs_T0)+1)+self.n_steps_human_task*self.n_human]))
                    print("          " + " \t basket_objs \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1)+self.n_steps_human_task*self.n_human:
                                                                          (len(self.BWPs_T0)+1)+self.n_steps_human_task*self.n_human+self.n_objs]))
                    if not self.obs_human_wait:
                        print("          " + " \t obj_ready \t\t{}".format(self.old_observations[idx][-2:]))
                    else:
                        print("          " + " \t obj_ready \t\t{}".format(self.old_observations[idx][-(self.n_human+2):-self.n_human]))
                        print("          " + " \t human_wait \t\t{}".format(self.old_observations[idx][-self.n_human:]))
                    print("")

                continue

            # get observation about location
            T_obs_0 = np.zeros(len(self.BWPs_T0)+1)
            if agent.cur_BWP is not None:
                T_obs_0[agent.cur_BWP.idx] = 1.0
            else:
                T_obs_0[-1] = 1.0
            BWP =agent.cur_BWP

            if self.rendering:
                print("Turtlebot" + str(idx) + " \t loc  \t\t\t{}".format(T_obs_0))

            # get observation about the human's current working step
            T_obs_1 = np.zeros(self.n_steps_human_task*self.n_human)
            if BWP is not None and BWP.idx < self.n_human:               #tweak depends on number of humans
                T_obs_1[self.humans[BWP.idx].cur_step + BWP.idx*self.n_steps_human_task] = 1.0

            if self.rendering:
                print("          " + " \t Hm0_cur_step \t\t{}".format(T_obs_1[0:self.n_steps_human_task]))
                print("          " + " \t Hm1_cur_step \t\t{}".format(T_obs_1[-self.n_steps_human_task:]))

            T_obs_3 = agent.objs_in_basket

            if self.rendering:
                print("          " + " \t Basket_objs \t\t{}".format(T_obs_3))

            # get observation about which turtlebot's tool is ready (This message only can be received in Tool Room)
            if BWP is None or BWP.idx >= len(self.BWPs_T0) - 2:
                T_obs_4 = self.agents[2].ready_objs
            else:
                T_obs_4 = np.zeros(2)

            if self.rendering:
                print("          " + " \t Obj_ready \t\t{}".format(T_obs_4))
                print("")

            # observe human waiting
            if self.obs_human_wait:
                T_obs_5 = np.zeros(self.n_human)
                for idx, human in enumerate(self.humans):
                    if human.cur_step_time_left < 0.0:
                        T_obs_5[idx] = 1
                if self.rendering:
                    print("          " + " \t human_wait \t\t{}".format(T_obs_5))


            # collect obs to be an array with shape (self.observation_space_T.n, )
            if not self.obs_human_wait:
                T_obs = np.hstack((T_obs_0, T_obs_1, T_obs_3, T_obs_4))
            else:
                T_obs = np.hstack((T_obs_0, T_obs_1, T_obs_3, T_obs_4, T_obs_5))
            assert len(T_obs) == self.observation_space_T.n

            observations.append(T_obs)
            self.old_observations[idx] = T_obs

        #--------------------get observations for Fetch robot
        if not self.agents[2].cur_action_done:
            observations.append(self.old_observations[2])
            if self.rendering:
                print("Fetchrobot" + " \t which_obj_ready  \t{}".format(self.old_observations[2][0:3]))
                print("          " + " \t T#_beside_table  \t{}".format(self.old_observations[2][3:5]))
                if self.obs_human_wait:
                    print("          " + " \t human_wait \t\t{}".format(self.old_observations[2][-self.n_human:]))
                print(" ")
                print("          " + " \t Found_objs  \t{}".format(self.agents[2].found_objs))
        else:
            # get observation about which objects are ready
            F_obs_0 = np.zeros(self.n_objs)
            for obj_idx in self.agents[2].found_objs:
                F_obs_0[obj_idx] += 1
            
            if self.rendering:
                print("Fetchrobot" + " \t which_obj_ready  \t{}".format(F_obs_0))

            # get observation about which turtlebot is beside the table
            F_obs_1 = np.zeros(2)
            for idx, agent in enumerate(self.agents[0:2]):
                if agent.xcoord == agent.BWPs[-1].xcoord and agent.ycoord == agent.BWPs[-1].ycoord:
                    F_obs_1[idx] = 1.0
            
            if self.rendering:
                print("          " + " \t T#_beside_table  \t{}".format(F_obs_1))

            # get observation about human waiting status
            if self.obs_human_wait:
                F_obs_2 = np.zeros(self.n_human)
                for idx, human in enumerate(self.humans):
                    if human.cur_step_time_left < 0.0:
                        F_obs_2[idx] = 1
                print("          " + " \t human_wait \t\t{}".format(F_obs_2))

            if self.rendering:
                print("          " + " \t Found_objs  \t{}".format(self.agents[2].found_objs))

            # collect obs to be an array with shape (self.observation_space_F.n, )
            if not self.obs_human_wait:
                F_obs = np.hstack((F_obs_0, F_obs_1))
            else:
                F_obs = np.hstack((F_obs_0, F_obs_1, F_obs_2))

            assert len(F_obs) == self.observation_space_F.n
            self.old_observations[2] = F_obs
            observations.append(F_obs)

        return observations

    def get_state(self):

        state = []
        # each agent's position
        for ag in self.agents:
            state.append(ag.xcoord / 7.0)
            state.append(ag.ycoord / 5.0)

        # objects in basket
        state += self.agents[0].objs_in_basket.tolist()
        state += self.agents[1].objs_in_basket.tolist()

        # which obj is in the staging area
        objs = np.zeros(self.n_objs)
        for obj_idx in self.agents[2].found_objs:
            objs[obj_idx] += 1
        state += objs.tolist()

        # whether fetch is passing any object
        obj_inhand = np.zeros(self.n_objs)
        if self.agents[2].cur_action is not None:
            if self.agents[2].cur_action.idx in [1,2] and \
                    not self.agents[2].cur_action_done and \
                    len(self.agents[2].found_objs) > 0:
                        obj_inhand[self.agents[2].found_objs[0]] = 1.0
        assert obj_inhand.sum() <= 1, "More than one object in fetch's hand ..."
        state += obj_inhand.tolist()

        # fetch's action status
        if self.agents[2].cur_action is not None:
            state.append(self.agents[2].cur_action_time_left / self.agents[2].cur_action.t_cost)
        else:
            state.append(0.0)

        # human status
        for human in self.humans:
            human_status = np.zeros(self.n_steps_human_task)
            human_status[human.cur_step] = 1.0
            state += human_status.tolist()
            if human.cur_step_time_left >= 0.0:
                state.append(1 - human.cur_step_time_left / \
                        human.expected_timecost_per_task_step[human.cur_step])
            else:
                state.append(-1.0)

        if self.rendering:
            print(f"Turtlebot1 pos {state[0]*7.0, state[1]*5.0}")
            print(f"Turtlebot2 pos {state[2]*7.0, state[3]*5.0}")
            print(f"Fetch pos {state[4]*7.0, state[5]*5.0}")
            print(f"Objs in Turtlebot1 basket {state[6:9]}")
            print(f"Objs in Turtlebot2 basket {state[9:12]}")
            print(f"Objs in staging area {state[12:15]}")
            print(f"Obj in hand {state[15:18]}")
            print(f"Fetch action status {state[18]}")
            print(f"Human_0 status {state[19:23]}")
            print(f"Human_0 current process {state[23]}")
            print(f"Human_1 status {state[24:28]}")
            print(f"Human_1 current process {state[28]}")

        return np.array(state)

    def render(self, mode='human'):

        mul = 2
        screen_width = 700 * mul
        screen_height = 500 * mul

        if self.viewer is None:
            from macro_marl.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            line = rendering.Line((0.0, 0.0), (0.0, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_height), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #--------------------------------draw rooms' boundaries

            for i in range(0,100,2):
                line_tool_room = rendering.Line((350*mul, i*5*mul), (350*mul, (i+1)*5*mul))
                line_tool_room.set_color(0,0,0)
                line_tool_room.linewidth.stroke = 2
                self.viewer.add_geom(line_tool_room)

            for i in range(0,80,2):
                line_WA = rendering.Line((500*mul+i*5*mul, 300*mul), (500*mul+(i+1)*5*mul, 300*mul))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,80,2):
                line_WA = rendering.Line((500*mul, 300*mul+i*5*mul), (500*mul, 300*mul+(i+1)*5*mul))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
                
            for i in range(0,80,2):
                line_WA = rendering.Line((500*mul+i*5*mul, 200*mul), (500*mul+(i+1)*5*mul, 200*mul))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,40,2):
                line_WA = rendering.Line((500*mul, 0+i*5*mul), (500*mul, 0+(i+1)*5*mul))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
            
            #---------------------------draw BW0
            for i in range(len(self.BWPs)):
                BWP = rendering.make_circle(radius=6*mul)
                BWP.set_color(178.0/255.0, 34.0/255.0, 34.0/255.0)
                BWPtrans = rendering.Transform(translation=(self.BWPs[i].xcoord*100*mul, self.BWPs[i].ycoord*100*mul))
                BWP.add_attr(BWPtrans)
                self.viewer.add_geom(BWP)

            #-------------------------------draw table
            tablewidth = 60.0*mul
            tableheight = 125.0*mul
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.43,0.28,0.02)
            tabletrans = rendering.Transform(translation=(175*mul, 250*mul))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            tablewidth = 54.0*mul
            tableheight = 119.0*mul
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.67,0.43,0.02)
            tabletrans = rendering.Transform(translation=(175*mul, 250*mul))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            #-----------------------------draw Fetch
            fetch = rendering.make_circle(radius=30*mul)
            fetch.set_color(*(0.0,0.0,0.0))
            self.fetchtrans = rendering.Transform(translation=(self.agents[2].xcoord*100*mul, self.agents[2].ycoord*100*mul))
            fetch.add_attr(self.fetchtrans)
            self.viewer.add_geom(fetch)

            #-----------------------------draw Fetch
            fetch_c = rendering.make_circle(radius=27*mul)
            fetch_c.set_color(*(0.5, 0.5,0.5))
            self.fetchtrans_c = rendering.Transform(translation=(self.agents[2].xcoord*100*mul, self.agents[2].ycoord*100*mul))
            fetch_c.add_attr(self.fetchtrans_c)
            self.viewer.add_geom(fetch_c)

            #-----------------------------draw Fetch arms
            self.arm2 = rendering.FilledPolygon([(-5.0*mul,-24.0*mul), (-5.0*mul, 24.0*mul), (5.0*mul, 24.0*mul), (5.0*mul, -24.0*mul)])
            self.arm2.set_color(0.0, 0.0, 0.0)
            self.arm2trans = rendering.Transform(translation=(self.agents[2].xcoord*10000*mul+55*mul, self.agents[2].ycoord*100*mul), rotation = -90/180*np.pi)
            self.arm2.add_attr(self.arm2trans)
            self.viewer.add_geom(self.arm2)

            self.arm2_c = rendering.FilledPolygon([(-3.0*mul,-22.0*mul), (-3.0*mul, 22.0*mul), (3.0*mul, 22.0*mul), (3.0*mul, -22.0*mul)])
            self.arm2_c.set_color(0.5, 0.5, 0.5)
            self.arm2trans_c = rendering.Transform(translation=(self.agents[2].xcoord*10000*mul+55*mul, self.agents[2].ycoord*100*mul), rotation = -90/180*np.pi)
            self.arm2_c.add_attr(self.arm2trans_c)
            self.viewer.add_geom(self.arm2_c)

            self.arm1 = rendering.FilledPolygon([(-5.0*mul,-36.0*mul), (-5.0*mul, 36.0*mul), (5.0*mul, 36.0*mul), (5.0*mul, -36.0*mul)])
            self.arm1.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108*mul, 187*mul), rotation = 15/180*np.pi)
            self.arm1.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1)

            self.arm1_c = rendering.FilledPolygon([(-3.0*mul,-34.0*mul), (-3.0*mul, 34.0*mul), (3.0*mul, 34.0*mul), (3.0*mul, -34.0*mul)])
            self.arm1_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108*mul, 187*mul), rotation = 15/180*np.pi)
            self.arm1_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1_c)
            
            self.arm0 = rendering.FilledPolygon([(-5.0*mul,-36.0*mul), (-5.0*mul, 36.0*mul), (5.0*mul, 36.0*mul), (5.0*mul, -36.0*mul)])
            self.arm0.set_color(1.0, 1.0, 1.0)
            arm0trans = rendering.Transform(translation=(108*mul, 313*mul), rotation = -15/180*np.pi)
            self.arm0.add_attr(arm0trans)
            self.viewer.add_geom(self.arm0)

            self.arm0_c = rendering.FilledPolygon([(-3.0*mul,-34.0*mul), (-3.0*mul, 34.0*mul), (3.0*mul, 34.0*mul), (3.0*mul, -34.0*mul)])
            self.arm0_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108*mul, 313*mul), rotation = -15/180*np.pi)
            self.arm0_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm0_c)

            #----------------------------draw Turtlebot_1
            turtlebot_1 = rendering.make_circle(radius=18.0*mul)
            turtlebot_1.set_color(*(0.15,0.65,0.15))
            self.turtlebot_1trans = rendering.Transform(translation=(self.agents[0].xcoord*100*mul, self.agents[0].ycoord*100*mul))
            turtlebot_1.add_attr(self.turtlebot_1trans)
            self.viewer.add_geom(turtlebot_1)

            turtlebot_1_c = rendering.make_circle(radius=16.0*mul)
            turtlebot_1_c.set_color(*(0.0,0.8,0.4))
            self.turtlebot_1trans_c = rendering.Transform(translation=(self.agents[0].xcoord*100*mul, self.agents[0].ycoord*100*mul))
            turtlebot_1_c.add_attr(self.turtlebot_1trans_c)
            self.viewer.add_geom(turtlebot_1_c)
            
            #----------------------------draw Turtlebot_2
            turtlebot_2 = rendering.make_circle(radius=18.0*mul)
            turtlebot_2.set_color(*(0.15,0.15,0.65))
            self.turtlebot_2trans = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100*mul))
            turtlebot_2.add_attr(self.turtlebot_2trans)
            self.viewer.add_geom(turtlebot_2)

            turtlebot_2_c = rendering.make_circle(radius=16.0*mul)
            turtlebot_2_c.set_color(*(0.0,0.4,0.8))
            self.turtlebot_2trans_c = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100*mul))
            turtlebot_2_c.add_attr(self.turtlebot_2trans_c)
            self.viewer.add_geom(turtlebot_2_c)

            #----------------------------draw human_0's status
            self.human0_progress_bar = []
            total_steps = self.humans[0].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10*mul,-10*mul), (-10*mul,10*mul), (10*mul,10*mul), (10*mul,-10*mul)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520*mul+i*26*mul,480*mul))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human0_progress_bar.append(progress_bar)
            
            #----------------------------draw human_1's status
            self.human1_progress_bar = []
            total_steps = self.humans[1].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10*mul,-10*mul), (-10*mul,10*mul), (10*mul,10*mul), (10*mul,-10*mul)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520*mul+i*26*mul,20*mul))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human1_progress_bar.append(progress_bar)
            

        # draw each robot's status
        self.turtlebot_1trans.set_translation(self.agents[0].xcoord*100*mul, self.agents[0].ycoord*100*mul)
        self.turtlebot_1trans_c.set_translation(self.agents[0].xcoord*100*mul, self.agents[0].ycoord*100*mul)
        self.turtlebot_2trans.set_translation(self.agents[1].xcoord*100*mul, self.agents[1].ycoord*100*mul)
        self.turtlebot_2trans_c.set_translation(self.agents[1].xcoord*100*mul, self.agents[1].ycoord*100*mul)
        self.fetchtrans.set_translation(self.agents[2].xcoord*100*mul, self.agents[2].ycoord*100*mul)

        # # reset human's progress bar
        for idx, bar in enumerate(self.human0_progress_bar):
            bar.set_color(0.8,0.8,0.8)
        for idx, bar in enumerate(self.human1_progress_bar):
            bar.set_color(0.8,0.8,0.8)

        # draw each human's status
        if self.humans[0].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx < self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[0].cur_step:
                    bar.set_color(*self.human_bar_colors[0])
                    break
        else:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx <= self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # draw each human's status
        if self.humans[1].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx < self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[1].cur_step:
                    bar.set_color(*self.human_bar_colors[1])
                    break
        else:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx <= self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # reset fetch arm
        self.arm0.set_color(1.0, 1.0, 1.0)
        self.arm0_c.set_color(1.0, 1.0, 1.0)
        self.arm1.set_color(1.0, 1.0, 1.0)
        self.arm1_c.set_color(1.0, 1.0, 1.0)

        self.arm2trans_c.set_translation(self.agents[2].xcoord*10000*mul+55*mul, self.agents[2].ycoord*100*mul)
        self.arm2trans.set_translation(self.agents[2].xcoord*10000*mul+55*mul, self.agents[2].ycoord*100*mul)


        self.pass_objs = 0

        if self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 1 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    self.pass_objs += 1
                    self.arm0.set_color(0.0, 0.0, 0.0)
                    self.arm0_c.set_color(0.5,0.5,0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 2 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    self.pass_objs += 1
                    self.arm1.set_color(0.0, 0.0, 0.0)
                    self.arm1_c.set_color(0.5, 0.5, 0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx > 2 and \
                np.sum(self.agents[2].count_found_obj) < self.n_objs*self.n_each_obj:
                    self.arm2trans_c.set_translation(self.agents[2].xcoord*100*mul+55*mul, self.agents[2].ycoord*100*mul)
                    self.arm2trans.set_translation(self.agents[2].xcoord*100*mul+55*mul, self.agents[2].ycoord*100*mul)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class ObjSearchDelivery_v8(ObjSearchDelivery_v7):

    """1) random sample fast human and slow human position"""

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, *args, **kwargs):

        super(ObjSearchDelivery_v8, self).__init__(*args, **kwargs)

    def create_turtlebot_actions(self):

        self.action_space_T = spaces.Discrete(4)

        if not self.obs_human_wait:
            # Descrete location areas: 5 
            # human's working step: 4*2
            # human's procress on the current step: 1
            # which object in the basket: n_objs
            # the number of objs are ready: 2
            self.observation_space_T = spaces.MultiBinary(5+self.n_steps_human_task*self.n_human + self.n_human +self.n_objs+2)
        else:
            self.observation_space_T = spaces.MultiBinary(5+self.n_steps_human_task*self.n_human + self.n_human  +self.n_objs+2+self.n_human)
        #-----------------def single step macro-actions for Turtlebot 
        T_MA0 = MacroAction('Go_WA0', 0, expected_t_cost = None, ma_bwpterm = 0)
        T_MA1 = MacroAction('Go_WA1', 1, expected_t_cost = None, ma_bwpterm = 1)
        T_MA2 = MacroAction('Go_Tool_Room', 2, expected_t_cost = None, ma_bwpterm = 2)
        T_MA3 = MacroAction('Get_Tool', 3, expected_t_cost = None, ma_bwpterm = 3)

        self.T_MAs = [T_MA0, T_MA1, T_MA2, T_MA3]

    def createHumans(self):

        if np.random.uniform() > 0.5:
            #-----------------initialize Two Humans
            Human0 = AgentHuman(0, self.n_steps_human_task, self.human_speed[0], list(range(self.n_objs)))
            Human1 = AgentHuman(1, self.n_steps_human_task, self.human_speed[1], list(range(self.n_objs)))
            self.human_bar_colors = [(1.0, 0.5, 0.0), (0.0, 1.0, 0.0)]
        else:
            Human0 = AgentHuman(0, self.n_steps_human_task, self.human_speed[1], list(range(self.n_objs)))
            Human1 = AgentHuman(1, self.n_steps_human_task, self.human_speed[0], list(range(self.n_objs)))
            self.human_bar_colors = [(0.0, 1.0, 0.0), (1.0, 0.5, 0.0)]

        self.humans = [Human0, Human1]
        self.n_human = len(self.humans)
        self.n_human_finished = []   # recording the number of human who has finished his own task

    def _getobs(self):

        #--------------------get observations at the beginning of each episode
        if self.t == 0:
            # get initial observation for turtlebot0
            T_obs_0 = np.zeros(self.observation_space_T.n)
            T_obs_0[len(self.BWPs_T0)] = 1.0

            # get initial observation for turtlebot1
            T_obs_1 = np.zeros(self.observation_space_T.n)
            T_obs_1[len(self.BWPs_T1)] = 1.0

            # get initial observaion for fetch robot
            F_obs = np.zeros(self.observation_space_F.n) 

            observations = [T_obs_0, T_obs_1, F_obs]
            self.t = 1
            self.old_observations = observations

            return observations

        #---------------------get observations for the two turtlebots
        if self.rendering:
            print("")
            print("observations list:")

        observations = []
        for idx, agent in enumerate(self.agents[0:2]):

            # won't get new obs until current macro-action finished
            if not agent.cur_action_done:
                observations.append(self.old_observations[idx])
                if self.rendering:
                    print("Turtlebot" + str(idx) + " \t loc  \t\t\t{}".format(self.old_observations[idx][0:(len(self.BWPs_T0)+1)]))
                    print("          " + " \t hm0_cur_step \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1):
                                                                          (len(self.BWPs_T0)+1)+self.n_steps_human_task]))
                    print("          " + " \t hm0_cur_proc \t\t{}".format(self.old_observations[idx][len(self.BWPs_T0) + \
                                                                          1 + \
                                                                          self.n_steps_human_task]))
                    print("          " + " \t hm1_cur_step \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1)+self.n_steps_human_task+1:
                                                                          (len(self.BWPs_T0)+1)+self.n_steps_human_task*self.n_human+1]))
                    print("          " + " \t hm1_cur_proc \t\t{}".format(self.old_observations[idx][len(self.BWPs_T0) + \
                                                                          1 + self.n_steps_human_task*self.n_human + \
                                                                          1]))
                    print("          " + " \t basket_objs \t\t{}".format(self.old_observations[idx][(len(self.BWPs_T0)+1)+self.n_steps_human_task*self.n_human+2:
                                                                          (len(self.BWPs_T0)+1)+self.n_steps_human_task*self.n_human+2+self.n_objs]))
                    if not self.obs_human_wait:
                        print("          " + " \t obj_ready \t\t{}".format(self.old_observations[idx][-2:]))
                    else:
                        print("          " + " \t obj_ready \t\t{}".format(self.old_observations[idx][-(self.n_human+2):-self.n_human]))
                        print("          " + " \t human_wait \t\t{}".format(self.old_observations[idx][-self.n_human:]))
                    print("")

                continue

            # get observation about location
            T_obs_0 = np.zeros(len(self.BWPs_T0)+1)
            if agent.cur_BWP is not None:
                T_obs_0[agent.cur_BWP.idx] = 1.0
            else:
                T_obs_0[-1] = 1.0
            BWP =agent.cur_BWP

            if self.rendering:
                print("Turtlebot" + str(idx) + " \t loc  \t\t\t{}".format(T_obs_0))

            # get observation about the human's current working step
            T_obs_1 = np.zeros(self.n_steps_human_task*self.n_human + self.n_human)
            if BWP is not None and BWP.idx < self.n_human:               #tweak depends on number of humans
                human = self.humans[BWP.idx]
                T_obs_1[human.cur_step + BWP.idx*(self.n_steps_human_task+1)] = 1.0
                if human.cur_step_time_left >= 0.0:
                    T_obs_1[BWP.idx*(self.n_steps_human_task+1) + self.n_steps_human_task] = 1 - \
                            human.cur_step_time_left / human.expected_timecost_per_task_step[human.cur_step]
                else:
                    T_obs_1[BWP.idx*(self.n_steps_human_task+1) + self.n_steps_human_task] = -1.0

            if self.rendering:
                print("          " + " \t Hm0_cur_step \t\t{}".format(T_obs_1[0:self.n_steps_human_task]))
                print("          " + " \t Hm0_cur_proc \t\t{}".format(T_obs_1[self.n_steps_human_task]))
                print("          " + " \t Hm1_cur_step \t\t{}".format(T_obs_1[-self.n_steps_human_task-1:-1]))
                print("          " + " \t Hm1_cur_proc \t\t{}".format(T_obs_1[-1]))

            T_obs_3 = agent.objs_in_basket

            if self.rendering:
                print("          " + " \t Basket_objs \t\t{}".format(T_obs_3))

            # get observation about the number of objs are ready in the staging area.
            if BWP is None or BWP.idx >= len(self.BWPs_T0) - 2:
                T_obs_4 = self.agents[2].ready_objs
            else:
                T_obs_4 = np.zeros(2)

            if self.rendering:
                print("          " + " \t Obj_ready \t\t{}".format(T_obs_4))
                print("")

            # observe human waiting
            if self.obs_human_wait:
                T_obs_5 = np.zeros(self.n_human)
                for idx, human in enumerate(self.humans):
                    if human.cur_step_time_left < 0.0:
                        T_obs_5[idx] = 1
                if self.rendering:
                    print("          " + " \t human_wait \t\t{}".format(T_obs_5))


            # collect obs to be an array with shape (self.observation_space_T.n, )
            if not self.obs_human_wait:
                T_obs = np.hstack((T_obs_0, T_obs_1, T_obs_3, T_obs_4))
            else:
                T_obs = np.hstack((T_obs_0, T_obs_1, T_obs_3, T_obs_4, T_obs_5))
            assert len(T_obs) == self.observation_space_T.n

            observations.append(T_obs)
            self.old_observations[idx] = T_obs

        #--------------------get observations for Fetch robot
        if not self.agents[2].cur_action_done:
            observations.append(self.old_observations[2])
            if self.rendering:
                print("Fetchrobot" + " \t which_obj_ready  \t{}".format(self.old_observations[2][0:3]))
                print("          " + " \t T#_beside_table  \t{}".format(self.old_observations[2][3:5]))
                if self.obs_human_wait:
                    print("          " + " \t human_wait \t\t{}".format(self.old_observations[2][-self.n_human:]))
                print(" ")
                print("          " + " \t Found_objs  \t{}".format(self.agents[2].found_objs))
        else:
            # get observation about which objects are ready
            F_obs_0 = np.zeros(self.n_objs)
            for obj_idx in self.agents[2].found_objs:
                F_obs_0[obj_idx] += 1
            
            if self.rendering:
                print("Fetchrobot" + " \t which_obj_ready  \t{}".format(F_obs_0))

            # get observation about which turtlebot is beside the table
            F_obs_1 = np.zeros(2)
            for idx, agent in enumerate(self.agents[0:2]):
                if agent.xcoord == agent.BWPs[-1].xcoord and agent.ycoord == agent.BWPs[-1].ycoord:
                    F_obs_1[idx] = 1.0
            
            if self.rendering:
                print("          " + " \t T#_beside_table  \t{}".format(F_obs_1))

            # get observation about human waiting status
            if self.obs_human_wait:
                F_obs_2 = np.zeros(self.n_human)
                for idx, human in enumerate(self.humans):
                    if human.cur_step_time_left < 0.0:
                        F_obs_2[idx] = 1
                print("          " + " \t human_wait \t\t{}".format(F_obs_2))

            if self.rendering:
                print("          " + " \t Found_objs  \t{}".format(self.agents[2].found_objs))

            # collect obs to be an array with shape (self.observation_space_F.n, )
            if not self.obs_human_wait:
                F_obs = np.hstack((F_obs_0, F_obs_1))
            else:
                F_obs = np.hstack((F_obs_0, F_obs_1, F_obs_2))

            assert len(F_obs) == self.observation_space_F.n
            self.old_observations[2] = F_obs
            observations.append(F_obs)

        return observations

    def reset(self):
        
        # reset the agents in this env
        self.createAgents()

        # reset the humans in this env
        self.createHumans()
        self.finish_serve_humans = []
        
        self.t = 0   # indicates the beginning of one episode, check _getobs()
        self.count_step = 0

        if self.rendering:
            self.render()

        return self._getobs()
 
    def render(self, mode='human'):

        screen_width = 700
        screen_height = 500

        if self.viewer is None:
            from macro_marl.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            line = rendering.Line((0.0, 0.0), (0.0, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_height), (screen_width, screen_height))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #--------------------------------draw rooms' boundaries

            for i in range(0,100,2):
                line_tool_room = rendering.Line((350, i*5), (350, (i+1)*5))
                line_tool_room.set_color(0,0,0)
                line_tool_room.linewidth.stroke = 2
                self.viewer.add_geom(line_tool_room)

            for i in range(0,80,2):
                line_WA = rendering.Line((500+i*5, 300), (500+(i+1)*5, 300))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,80,2):
                line_WA = rendering.Line((500, 300+i*5), (500, 300+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
                
            for i in range(0,80,2):
                line_WA = rendering.Line((500+i*5, 200), (500+(i+1)*5, 200))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)

            for i in range(0,40,2):
                line_WA = rendering.Line((500, 0+i*5), (500, 0+(i+1)*5))
                line_WA.linewidth.stroke = 2
                line_WA.set_color(0,0,0)
                self.viewer.add_geom(line_WA)
            
            #---------------------------draw BW0
            for i in range(len(self.BWPs)):
                BWP = rendering.make_circle(radius=6)
                BWP.set_color(178.0/255.0, 34.0/255.0, 34.0/255.0)
                BWPtrans = rendering.Transform(translation=(self.BWPs[i].xcoord*100, self.BWPs[i].ycoord*100))
                BWP.add_attr(BWPtrans)
                self.viewer.add_geom(BWP)

            #-------------------------------draw table
            tablewidth = 60.0
            tableheight = 125.0
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.43,0.28,0.02)
            tabletrans = rendering.Transform(translation=(175, 250))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            tablewidth = 54.0
            tableheight = 119.0
            l,r,t,b = -tablewidth/2.0, tablewidth/2.0, tableheight/2.0, -tableheight/2.0
            table = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            table.set_color(0.67,0.43,0.02)
            tabletrans = rendering.Transform(translation=(175, 250))
            table.add_attr(tabletrans)
            self.viewer.add_geom(table)

            #-----------------------------draw Fetch
            fetch = rendering.make_circle(radius=26)
            fetch.set_color(*(0.0,0.0,0.0))
            self.fetchtrans = rendering.Transform(translation=(self.agents[2].xcoord*100, self.agents[2].ycoord*100))
            fetch.add_attr(self.fetchtrans)
            self.viewer.add_geom(fetch)

            #-----------------------------draw Fetch
            fetch_c = rendering.make_circle(radius=23)
            fetch_c.set_color(*(0.5, 0.5,0.5))
            self.fetchtrans_c = rendering.Transform(translation=(self.agents[2].xcoord*100, self.agents[2].ycoord*100))
            fetch_c.add_attr(self.fetchtrans_c)
            self.viewer.add_geom(fetch_c)

            #-----------------------------draw Fetch arms
            self.arm2 = rendering.FilledPolygon([(-5.0,-20.0,), (-5.0, 20.0), (5.0, 20.0), (5.0, -20.0)])
            self.arm2.set_color(0.0, 0.0, 0.0)
            self.arm2trans = rendering.Transform(translation=(self.agents[2].xcoord*10000+49, self.agents[2].ycoord*100), rotation = -90/180*np.pi)
            self.arm2.add_attr(self.arm2trans)
            self.viewer.add_geom(self.arm2)

            self.arm2_c = rendering.FilledPolygon([(-3.0,-18.0,), (-3.0, 18.0), (3.0, 18.0), (3.0, -18.0)])
            self.arm2_c.set_color(0.5, 0.5, 0.5)
            self.arm2trans_c = rendering.Transform(translation=(self.agents[2].xcoord*10000+48, self.agents[2].ycoord*100), rotation = -90/180*np.pi)
            self.arm2_c.add_attr(self.arm2trans_c)
            self.viewer.add_geom(self.arm2_c)

            self.arm1 = rendering.FilledPolygon([(-5.0,-38.0,), (-5.0, 38.0), (5.0, 38.0), (5.0, -38.0)])
            self.arm1.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 187), rotation = 15/180*np.pi)
            self.arm1.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1)

            self.arm1_c = rendering.FilledPolygon([(-3.0,-36.0,), (-3.0, 36.0), (3.0, 36.0), (3.0, -36.0)])
            self.arm1_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 187), rotation = 15/180*np.pi)
            self.arm1_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm1_c)
            
            self.arm0 = rendering.FilledPolygon([(-5.0,-38.0,), (-5.0, 38.0), (5.0, 38.0), (5.0, -38.0)])
            self.arm0.set_color(1.0, 1.0, 1.0)
            arm0trans = rendering.Transform(translation=(108, 313), rotation = -15/180*np.pi)
            self.arm0.add_attr(arm0trans)
            self.viewer.add_geom(self.arm0)

            self.arm0_c = rendering.FilledPolygon([(-3.0,-36.0,), (-3.0, 36.0), (3.0, 36.0), (3.0, -36.0)])
            self.arm0_c.set_color(1.0, 1.0, 1.0)
            arm1trans = rendering.Transform(translation=(108, 313), rotation = -15/180*np.pi)
            self.arm0_c.add_attr(arm1trans)
            self.viewer.add_geom(self.arm0_c)

            #----------------------------draw Turtlebot_1
            turtlebot_1 = rendering.make_circle(radius=17.0)
            turtlebot_1.set_color(*(0.15,0.65,0.15))
            self.turtlebot_1trans = rendering.Transform(translation=(self.agents[0].xcoord*100, self.agents[0].ycoord*100))
            turtlebot_1.add_attr(self.turtlebot_1trans)
            self.viewer.add_geom(turtlebot_1)

            turtlebot_1_c = rendering.make_circle(radius=14.0)
            turtlebot_1_c.set_color(*(0.0,0.8,0.4))
            self.turtlebot_1trans_c = rendering.Transform(translation=(self.agents[0].xcoord*100, self.agents[0].ycoord*100))
            turtlebot_1_c.add_attr(self.turtlebot_1trans_c)
            self.viewer.add_geom(turtlebot_1_c)
            
            #----------------------------draw Turtlebot_2
            turtlebot_2 = rendering.make_circle(radius=17.0)
            turtlebot_2.set_color(*(0.15,0.15,0.65))
            self.turtlebot_2trans = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100))
            turtlebot_2.add_attr(self.turtlebot_2trans)
            self.viewer.add_geom(turtlebot_2)

            turtlebot_2_c = rendering.make_circle(radius=14.0)
            turtlebot_2_c.set_color(*(0.0,0.4,0.8))
            self.turtlebot_2trans_c = rendering.Transform(translation=(self.agents[1].xcoord*100, self.agents[1].ycoord*100))
            turtlebot_2_c.add_attr(self.turtlebot_2trans_c)
            self.viewer.add_geom(turtlebot_2_c)

            #----------------------------draw human_0's status
            self.human0_progress_bar = []
            total_steps = self.humans[0].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10,-10), (-10,10), (10,10), (10,-10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26,480))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human0_progress_bar.append(progress_bar)
            
            #----------------------------draw human_1's status
            self.human1_progress_bar = []
            total_steps = self.humans[1].task_total_steps
            for i in range(total_steps):
                progress_bar = rendering.FilledPolygon([(-10,-10), (-10,10), (10,10), (10,-10)])
                progress_bar.set_color(0.8, 0.8, 0.8)
                progress_bartrans = rendering.Transform(translation=(520+i*26,20))
                progress_bar.add_attr(progress_bartrans)
                self.viewer.add_geom(progress_bar)
                self.human1_progress_bar.append(progress_bar)
            

        # draw each robot's status
        self.turtlebot_1trans.set_translation(self.agents[0].xcoord*100, self.agents[0].ycoord*100)
        self.turtlebot_1trans_c.set_translation(self.agents[0].xcoord*100, self.agents[0].ycoord*100)
        self.turtlebot_2trans.set_translation(self.agents[1].xcoord*100, self.agents[1].ycoord*100)
        self.turtlebot_2trans_c.set_translation(self.agents[1].xcoord*100, self.agents[1].ycoord*100)
        self.fetchtrans.set_translation(self.agents[2].xcoord*100, self.agents[2].ycoord*100)

        # # reset human's progress bar
        for idx, bar in enumerate(self.human0_progress_bar):
            bar.set_color(0.8,0.8,0.8)
        for idx, bar in enumerate(self.human1_progress_bar):
            bar.set_color(0.8,0.8,0.8)

        # draw each human's status
        if self.humans[0].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx < self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[0].cur_step:
                    bar.set_color(*self.human_bar_colors[0])
                    break
        else:
            for idx, bar in enumerate(self.human0_progress_bar):
                if idx <= self.humans[0].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # draw each human's status
        if self.humans[1].cur_step_time_left > 0:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx < self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)
                if idx == self.humans[1].cur_step:
                    bar.set_color(*self.human_bar_colors[1])
                    break
        else:
            for idx, bar in enumerate(self.human1_progress_bar):
                if idx <= self.humans[1].cur_step:
                    bar.set_color(0.0,0.0,0.0)
        
        # reset fetch arm
        self.arm0.set_color(1.0, 1.0, 1.0)
        self.arm0_c.set_color(1.0, 1.0, 1.0)
        self.arm1.set_color(1.0, 1.0, 1.0)
        self.arm1_c.set_color(1.0, 1.0, 1.0)

        self.arm2trans_c.set_translation(self.agents[2].xcoord*10000+48, self.agents[2].ycoord*100)
        self.arm2trans.set_translation(self.agents[2].xcoord*10000+49, self.agents[2].ycoord*100)


        self.pass_objs = 0

        if self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 1 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    self.pass_objs += 1
                    self.arm0.set_color(0.0, 0.0, 0.0)
                    self.arm0_c.set_color(0.5,0.5,0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx == 2 and \
                self.agents[2].cur_action_time_left <= 0.0 and \
                not self.agents[2].serving_failed and self.pass_objs < self.n_objs:
                    self.pass_objs += 1
                    self.arm1.set_color(0.0, 0.0, 0.0)
                    self.arm1_c.set_color(0.5, 0.5, 0.5)

        elif self.agents[2].cur_action is not None and \
                self.agents[2].cur_action.idx > 2 and \
                np.sum(self.agents[2].count_found_obj) < self.n_objs*self.n_each_obj:
                    self.arm2trans_c.set_translation(self.agents[2].xcoord*100+48, self.agents[2].ycoord*100)
                    self.arm2trans.set_translation(self.agents[2].xcoord*100+49, self.agents[2].ycoord*100)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
