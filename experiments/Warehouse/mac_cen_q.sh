#/bin/bash

# Warehouse-A
for ((i=0; i<20; i++))
do
     ma_cen_condi_ddrqn.py  --save_dir='ma_cen_condi_ddrqn_warehouse_A' \
                                --env_id='OSD-D-v7' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --batch_size=64 \
                                --train_freq=128 \
                                --total_epi=40_000 \
                                --replay_buffer_size=2000 \
                                --l_rate=0.0001 \
                                --h_stable_at=10_000 \
                                --eps_l_d_steps=10_000 \
                                --eps_end=0.05 \
                                --discount=1.0 \
                                --start_train=2000 \
                                --rnn_h_size=64 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --sample_epi \
                                --h0_speed_ps 27 20 20 20 \
                                --h1_speed_ps 27 20 20 20 \
                                --d_pen=-20.0 \
                                --tb_m_speed=0.8 \
                                --run_time=240 \
                                --run_id=$i & 
done

# Warehouse-B
for ((i=0; i<20; i++))
do
     ma_cen_condi_ddrqn.py  --save_dir='ma_cen_condi_ddrqn_warehouse_B' \
                                --env_id='OSD-T-v0' \
                                --n_agent=3 \
                                --env_terminate_step=250 \
                                --batch_size=64 \
                                --train_freq=64 \
                                --total_epi=80_000 \
                                --replay_buffer_size=2000 \
                                --l_rate=0.00005 \
                                --h_stable_at=10_000 \
                                --eps_l_d_steps=10_000 \
                                --eps_end=0.05 \
                                --discount=1.0 \
                                --start_train=2000 \
                                --rnn_h_size=64 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --sample_epi \
                                --h0_speed_ps 40 40 40 40 \
                                --h1_speed_ps 40 40 40 40 \
                                --h2_speed_ps 40 40 40 40 \
                                --d_pen=-20.0 \
                                --tb_m_speed=0.8 \
                                --run_time=240 \
                                --run_id=$i & 
done

# Warehouse-E
for ((i=0; i<20; i++))
do
     ma_cen_condi_ddrqn.py  --save_dir='ma_cen_condi_ddrqn_warehouse_E' \
                                --env_id='OSD-D-v7' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --batch_size=64 \
                                --train_freq=128 \
                                --total_epi=40_000 \
                                --replay_buffer_size=2000 \
                                --l_rate=0.0001 \
                                --h_stable_at=10_000 \
                                --eps_l_d_steps=10_000 \
                                --eps_end=0.05 \
                                --discount=1.0 \
                                --start_train=2000 \
                                --rnn_h_size=64 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --sample_epi \
                                --h0_speed_ps 18 15 15 15 \
                                --h1_speed_ps 48 18 15 15 \
                                --d_pen=-20.0 \
                                --tb_m_speed=0.8 \
                                --run_time=240 \
                                --run_id=$i & 
done
