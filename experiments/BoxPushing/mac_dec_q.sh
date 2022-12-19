#/bin/bash

for ((i=0; i<20; i++))
do
     ma_hddrqn.py  --save_dir='ma_hddrqn_bp8' \
                                --env_id='BP-MA-v0' \
                                --grid_dim 8 8 \
                                --n_agent=2 \
                                --env_terminate_step=100 \
                                --batch_size=64 \
                                --train_freq=10 \
                                --total_epi=40_000 \
                                --replay_buffer_size=100_000 \
                                --trace_len=10 \
                                --l_rate=0.001 \
                                --h_stable_at=4_000 \
                                --eps_l_d_steps=4_000 \
                                --eps_end=0.05 \
                                --discount=0.98 \
                                --start_train=500 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done

for ((i=0; i<20; i++))
do
     ma_hddrqn.py  --save_dir='ma_hddrqn_bp10' \
                                --env_id='BP-MA-v0' \
                                --grid_dim 10 10 \
                                --n_agent=2 \
                                --env_terminate_step=100 \
                                --batch_size=32 \
                                --train_freq=14 \
                                --total_epi=40_000 \
                                --replay_buffer_size=100_000 \
                                --trace_len=14 \
                                --l_rate=0.001 \
                                --h_stable_at=6_000 \
                                --eps_l_d_steps=6_000 \
                                --eps_end=0.05 \
                                --discount=0.98 \
                                --start_train=500 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done

for ((i=0; i<20; i++))
do
     ma_hddrqn.py  --save_dir='ma_hddrqn_bp12' \
                                --env_id='BP-MA-v0' \
                                --grid_dim 12 12 \
                                --n_agent=2 \
                                --env_terminate_step=100 \
                                --batch_size=32 \
                                --train_freq=20 \
                                --total_epi=40_000 \
                                --replay_buffer_size=100_000 \
                                --trace_len=20 \
                                --l_rate=0.001 \
                                --h_stable_at=6_000 \
                                --eps_l_d_steps=6_000 \
                                --eps_end=0.05 \
                                --discount=0.98 \
                                --start_train=500 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done

for ((i=0; i<20; i++))
do
     ma_hddrqn.py  --save_dir='ma_hddrqn_bp20' \
                                --env_id='BP-MA-v0' \
                                --grid_dim 20 20 \
                                --n_agent=2 \
                                --env_terminate_step=200 \
                                --batch_size=32 \
                                --train_freq=35 \
                                --total_epi=40_000 \
                                --replay_buffer_size=100_000 \
                                --trace_len=35 \
                                --l_rate=0.001 \
                                --h_stable_at=8_000 \
                                --eps_l_d_steps=8_000 \
                                --eps_end=0.05 \
                                --discount=0.98 \
                                --start_train=500 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done

for ((i=0; i<20; i++))
do
     ma_hddrqn.py  --save_dir='ma_hddrqn_bp30' \
                                --env_id='BP-MA-v0' \
                                --grid_dim 30 30 \
                                --n_agent=2 \
                                --env_terminate_step=200 \
                                --batch_size=32 \
                                --train_freq=45 \
                                --total_epi=40_000 \
                                --replay_buffer_size=100_000 \
                                --trace_len=45 \
                                --l_rate=0.0005 \
                                --h_stable_at=8_000 \
                                --eps_l_d_steps=8_000 \
                                --eps_end=0.05 \
                                --discount=0.98 \
                                --start_train=500 \
                                --rnn \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done
