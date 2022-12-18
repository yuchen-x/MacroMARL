#!/usr/bin/python

from gym.envs.registration import register

register(
    id='BP-v0',
    entry_point='macro_marl.my_env.box_pushing:BoxPushing',
)

register(
    id='BP-MA-v0',
    entry_point='macro_marl.my_env.box_pushing_MA:BoxPushing_harder',
)

register(
    id='OSD-S-v4',
    entry_point='macro_marl.my_env.osd_ma_single_room:ObjSearchDelivery_v4',
)

register(
    id='OSD-D-v7',
    entry_point='macro_marl.my_env.osd_ma_double_room:ObjSearchDelivery_v7',
)

register(
    id='OSD-D-v8',
    entry_point='macro_marl.my_env.osd_ma_double_room:ObjSearchDelivery_v8',
)

register(
    id='OSD-T-v0',
    entry_point='macro_marl.my_env.osd_ma_tripple_room:ObjSearchDelivery_v0',
)

register(
    id='OSD-T-v1',
    entry_point='macro_marl.my_env.osd_ma_tripple_room:ObjSearchDelivery_v1',
)

register(
    id='OSD-F-v0',
    entry_point='macro_marl.my_env.osd_ma_four_room:ObjSearchDelivery_v0',
)
