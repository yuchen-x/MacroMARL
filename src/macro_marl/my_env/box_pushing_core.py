#!/usr/bin/python

import numpy as np
import gym
import IPython

from gym.utils import seeding

DIRECTION = np.array([[0.0, 1.0],
                      [1.0, 0.0],
                      [0.0, -1.0],
                      [-1.0, 0.0]])

class Agent(object):

    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 init_ori,
                 grid_dim,
                 penalty=-5,
                 speed=1.0):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.ori = init_ori
        self.direct = np.array([0.0,0.0])

        self.cur_action = None
        self.speed = speed

        self.xlen, self.ylen = grid_dim
        self.penalty = penalty
        self.pushing_big_box = False

    def step(self, action, boxes):

        assert action < 4, "The action received is out of range"

        reward = 0.0

        self.cur_action = action

        if action == 0:
            move = DIRECTION[self.ori]
            self.xcoord += move[0]
            self.ycoord += move[1]

            # check if touch the wall
            if self.xcoord > self.xlen-0.5 or \
               self.xcoord < 0.5 or \
               self.ycoord > self.ylen-0.5 or \
               self.ycoord < 0.5:
                   self.xcoord -= move[0]
                   self.ycoord -= move[1]
                   reward += self.penalty
            # check if push small box
            for box in boxes:
                if (box.xcoord == self.xcoord or \
                    box.xcoord == self.xcoord-0.5 or \
                    box.xcoord == self.xcoord+0.5) and box.ycoord == self.ycoord:
                    if self.ori == 0 and box.idx != 2:
                        box.xcoord += move[0]
                        box.ycoord += move[1]
                    else:
                        self.xcoord -= move[0]
                        self.ycoord -= move[1]
                        reward += self.penalty
        elif action == 1:
            if self.ori == 0:
                self.ori = 3
            else:
                self.ori -= 1

        elif action == 2:
            if self.ori == 3:
                self.ori = 0
            else:
                self.ori += 1

        return reward


class Box(object):
    
    def __init__(self,
                 idx,
                 init_x,
                 init_y,
                 size_h,
                 size_w):

        self.idx = idx
        self.xcoord = init_x
        self.ycoord = init_y
        self.h = size_h
        self.w = size_w






