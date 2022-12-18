#!/usr/bin/python

import numpy as np
import gym
import IPython

from gym.utils import seeding
from IPython.core.debugger import set_trace

class AgentTurtlebot_v4(object):

    """Properties for a Turtlebot"""

    def __init__(self, 
                 idx, 
                 init_x, 
                 init_y,
                 beliefwaypoints,
                 MAs,
                 n_objs,
                 speed = 0.6,
                 move_noisy=0.0,
                 get_tool_wait=10,
                 delay_delivery_penalty=0.0):

        # unique agent's id
        self.idx = idx
        # agent's name
        self.name = 'Turtlebot'+ str(self.idx)
        # agent's 2D position x 
        self.xcoord = init_x
        # agent's 2D position y
        self.ycoord = init_y
        # applicable waypoints to move to
        self.BWPs = beliefwaypoints
        # record which belief waypoint the agent currently is
        self.cur_BWP = None
        # obtain applicable macro_actions
        self.macro_actions = MAs
        # agent's current macro_action 
        self.cur_action = None
        # how much time left to finish current macro_action
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        # turtlebot base movement speed
        self.speed = speed
        self.move_noisy = move_noisy

        # communication info
        self.n_objs = n_objs
        # keep tracking the objects in the basket
        self.objs_in_basket = np.zeros(n_objs)
        # keep tracking the message of request objects received by Fetch robot
        self.request_objs = np.zeros(n_objs)
        self.get_tool_wait = get_tool_wait
        self.delay_delivery_penalty = delay_delivery_penalty

    def step(self, action, humans):

        """Depends on the input macro-action to run low-level controller to achieve 
           primitive action execution.
        """

        assert action < len(self.macro_actions), "The action received is out of the range"

        reward = 0.0

        # update current action info
        self.cur_action = self.macro_actions[action]
        self.cur_action_done = False

        # action 0 - 2
        if action <= 2:
            bwpterm_idx = self.cur_action.ma_bwpterm
            if self.cur_action.expected_t_cost != 1:
                dist = round(self._get_dist(self.BWPs[bwpterm_idx].xcoord, self.BWPs[bwpterm_idx].ycoord),2)
                if dist <= self.speed:
                    self.xcoord = self.BWPs[bwpterm_idx].xcoord
                    self.ycoord = self.BWPs[bwpterm_idx].ycoord
                    self.cur_BWP = self.BWPs[bwpterm_idx]
                    if self.cur_action_time_left > 0.0:
                        self.cur_action_time_left = 0.0
                        if action == 2:
                            return reward
                    if action < 2:
                        self.cur_action_done = True
                    else:
                        # indicates turtlebot has been ready to get obj from fetch
                        self.cur_action_time_left -= 1.0
                        if self.cur_action_time_left < -self.get_tool_wait:   #----get_tool action automatically terminate after waiting for 10s
                            self.cur_action_time_left = -1.0
                            self.cur_action_done = True
                else:
                    delta_x = self.speed / dist * (self.BWPs[bwpterm_idx].xcoord - self.xcoord) + np.random.normal(0.0, self.move_noisy)
                    delta_y = self.speed / dist * (self.BWPs[bwpterm_idx].ycoord - self.ycoord) + np.random.normal(0.0, self.move_noisy)
                    self.xcoord += delta_x
                    self.ycoord += delta_y
                    self.cur_action_time_left = dist - self.speed
            else:
                self.xcoord = self.BWPs[bwpterm_idx].xcoord
                self.ycoord = self.BWPs[bwpterm_idx].ycoord
                self.cur_BWP = self.BWPs[bwpterm_idx]

        # action 3 
        elif action >= 3:
            if self.xcoord < 3.5:
                self.request_objs[action-3] = 1.0
            self.cur_action_done = True

        # change the human's properties when turtlebot deliever correct objects
        if self.cur_BWP is not None and \
           (action == 0 and self.cur_BWP.idx == action):
            human = humans[self.cur_BWP.idx]
            if not human.next_requested_obj_obtained and \
                    self.objs_in_basket[human.next_request_obj_idx] > 0.0:
                        self.objs_in_basket[human.next_request_obj_idx] -= 1.0
                        if human.cur_step_time_left < 0 and self.delay_delivery_penalty:
                            reward += (100 + self.delay_delivery_penalty)
                        else:
                            reward += 100
                        human.next_requested_obj_obtained = True

        return reward

    def _get_dist(self, g_xcoord, g_ycoord):
        return np.sqrt((g_xcoord - self.xcoord)**2 + (g_ycoord - self.ycoord)**2)

class AgentFetch_v4(object):

    """Properties for a Fetch robot"""
    """Double Check for passing obj action, beginning and end"""

    def __init__(self, 
                 idx, 
                 init_x, 
                 init_y,
                 MAs,
                 n_objs,
                 n_each_obj,
                 manip_noisy,
                 drop_obj_penalty):

        # unique agent's id
        self.idx = idx
        # agent's name
        self.name = 'Fetch'
        # agent's 2D position x
        self.xcoord = init_x
        # agent's 2D position y
        self.ycoord = init_y
        # obtain applicable macro_actions
        self.macro_actions = MAs
        # manipulation noise on time-cost
        self.manip_noisy = manip_noisy
        # agent's current macro_action
        self.cur_action = None
        # how much time left to finish current macro_action
        self.cur_action_time_left = 0.0
        self.cur_action_done = True
        # the number of different objects in this env
        self.n_objs = n_objs
        # the amout of each obj in the env
        self.n_each_obj = n_each_obj
        self.count_found_obj = np.zeros(n_objs)
        # penalty for dropping object
        self.drop_obj_penalty = drop_obj_penalty
        
        ################# communication info ######################
        # indicates if fetch is serving or not
        self.serving = False   
        self.serving_failed = False
        # [0,0] means there is no any object ready for Turtlebot1 and Turtlebot2
        self.ready_objs = np.zeros(2)
        self.found_objs = []

    def step(self, action, agents):

        """Depends on the input macro-action to run low-level controller to achieve 
           primitive action execution.
        """

        reward = 0.0

        self.cur_action_time_left -= 1.0
        
        if self.cur_action_time_left  > 0.0:
            return reward
        else:
            if self.cur_action_done:
                self.cur_action = self.macro_actions[action]
                if self.manip_noisy:
                    noise = np.random.choice([-1, 0, 1, 2])
                    self.cur_action_time_left = self.cur_action.t_cost + noise - 1.0
                else:
                    self.cur_action_time_left = self.cur_action.t_cost - 1.0
                # action 0 wait request
                if self.cur_action.idx == 0:
                    self.cur_action_done = True
                else:
                    self.cur_action_done = False

                # when fetch execute pass_obj action, the corresponding turtlebot has to have been beside table
                if self.cur_action.idx == 1:
                    self.serving = True
                    if agents[0].cur_BWP is None or \
                       agents[0].cur_BWP.name != "ToolRoomTable" or \
                       agents[0].cur_action_time_left > -1.0:
                        self.serving_failed = True
                elif self.cur_action.idx == 2:
                    self.serving = True
                    if agents[1].cur_BWP is None or \
                       agents[1].cur_BWP.name != "ToolRoomTable" or \
                       agents[1].cur_action_time_left > -1.0:
                        self.serving_failed = True

                return reward

            # action 1 Pass_obj_T0
            elif self.cur_action.idx == 1:
                self.serving = False
                if not self.serving_failed and \
                   agents[0].cur_action_time_left < 0.0 and \
                   agents[0].cur_action.name == "Get_Tool":

                    if len(self.found_objs) > 0:
                        obj_idx = self.found_objs.pop(0)
                        agents[0].objs_in_basket[obj_idx] += 1.0
                    agents[0].cur_action_done = True
                    agents[0].cur_action_time_left = 0.0
                        
                    # check if there is still any other object ready for turtlebot 1
                    self.ready_objs = np.zeros(2)
                    if len(self.found_objs) == 1:
                        self.ready_objs[0]=1.0

                else:
                    reward += self.drop_obj_penalty

                self.serving_failed = False

            # action 2 Pass_obj_T1
            elif self.cur_action.idx == 2:
                self.serving = False
                if not self.serving_failed and \
                   agents[1].cur_action_time_left < 0.0 and \
                   agents[1].cur_action.name == "Get_Tool":

                    if len(self.found_objs) > 0:
                        obj_idx = self.found_objs.pop(0)
                        agents[1].objs_in_basket[obj_idx] += 1.0
                    agents[1].cur_action_done = True
                    agents[1].cur_action_time_left = 0.0

                    # check if there is still any other object ready for turtlebot 1
                    self.ready_objs = np.zeros(2)
                    if len(self.found_objs) == 1:
                        self.ready_objs[0] = 1.0

                else:
                    reward += self.drop_obj_penalty
                
                self.serving_failed = False

            # action Look_for_T0_obj
            elif self.cur_action.idx < 3+self.n_objs: 
                found_obj_idx = self.cur_action.idx - 3
                if len(self.found_objs) < 2 and self.count_found_obj[found_obj_idx] < self.n_each_obj:   #---------------tweak 3
                    self.count_found_obj[found_obj_idx] += 1.0
                    self.found_objs.append(found_obj_idx)
                    if len(self.found_objs) == 2:
                        self.ready_objs[1] = 1.0
                        self.ready_objs[0] = 0.0
                    else:
                        self.ready_objs[0] = 1.0
                        self.ready_objs[1] = 0.0

            # indicate the current action finished
            self.cur_action_done = True

        return reward

class AgentHuman(object):

    """Properties for a Human in the env"""

    def __init__(self,
                 idx,
                 task_total_steps,
                 expected_timecost_per_task_step,
                 request_objs_per_task_step,
                 std=None,
                 seed=None):

        # unique agent's id
        self.idx = idx
        # the total number of steps for finishing the task
        self.task_total_steps = task_total_steps
        # a vector to indicate the expected time cost for each human to finish each task step
        self.expected_timecost_per_task_step = expected_timecost_per_task_step
        # std is used to sample the actual time cost for each human to finish each task step
        self.time_cost_std_per_task_step = std
        # a vector to inidcate the tools needed for each task step
        self.request_objs_per_task_step = request_objs_per_task_step

        self.cur_step = 0 
        if std is None:
            self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
        else:
            # sample the time cost for the current task step, which will be counted down step by step
            self.cur_step_time_left = self.np.random.normal(self.expected_timecost_per_task_step[self.cur_step], self.time_cost_std_per_task_step)

        # indicates the tool needed for next task step
        self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step]
        # indicates if the tool needed for next step has been delivered
        self.next_requested_obj_obtained = False
        # indicates if the human has finished the whole task
        self.whole_task_finished = False

    def step(self):

        # check if the human already finished whole task
        if self.cur_step + 1 == self.task_total_steps:
            assert self.whole_task_finished == False
            self.whole_task_finished = True
        else:
            self.cur_step += 1
            if self.time_cost_std_per_task_step is None:
                self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
            else:
                self.cur_step_time_left = self.np.random.normal(self.expected_timecost_per_task_step[self.cur_step], self.time_cost_std_per_task_step)
            # update the request obj for next step
            if self.cur_step + 1 < self.task_total_steps:
                self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step] 
                self.next_requested_obj_obtained = False

    def reset(self):
        self.cur_step = 0
        if self.time_cost_std_per_task_step is None:
            self.cur_step_time_left = self.expected_timecost_per_task_step[self.cur_step]
        else:
            # sample the time cost for the current task step, which will be counted down step by step
            self.cur_step_time_left = self.np.random.normal(self.expected_timecost_per_task_step[self.cur_step], self.time_cost_std_per_task_step)

        # indicates the tool needed for next task step
        self.next_request_obj_idx = self.request_objs_per_task_step[self.cur_step]  
        # indicates if the tool needed for next step has been delivered
        self.next_requested_obj_obtained = False
        # indicates if the human has finished the whole task
        self.whole_task_finished = False

class MacroAction(object):

    """Properties for a macro_action"""

    def __init__(self, 
                 name,
                 idx,
                 expected_t_cost=None,
                 std=None,
                 ma_bwpterm=None):

        # the name of this macro-action
        self.name = name
        # the index of this macro-action
        self.idx = idx    
        # None is for moving action. When it is done depends on the specify speed.
        self.expected_t_cost = expected_t_cost
        self.std = std
        if std is None:
            # the time cost of finishing this macro-action
            self.real_t_cost = expected_t_cost
        else:
            self.real_t_cost = np.random.normal(expected_t_cost, std)
        # used for moving action to indicate at which belief waypoint this macro-action will be terminated,
        # None means the terminate belief waypoint is same as where the action is initialized.
        self.ma_bwpterm = ma_bwpterm

    @property
    def t_cost(self):
        if self.std is None:
            # the time cost of finishing this macro-action
            return self.expected_t_cost
        else:
            # resample a time cost for the macro-action
            return round(np.random.normal(self.expected_t_cost, self.std),1)
 
class BeliefWayPoint(object):

    """Properties for a waypoint in the 2D sapce"""

    def __init__(self,
                 name,
                 idx,
                 xcoord,
                 ycoord):
        
        self.name = name
        self.idx = idx
        self.xcoord = xcoord
        self.ycoord = ycoord

