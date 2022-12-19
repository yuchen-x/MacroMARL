#/bin/bash

# Box Pushing 6x6
for ((i=0; i<20; i++))
do
    ma_niacc_rnn_sV.py --save_dir='ma_niacc_bp6' \
                        --l_mode=0 \
                        --env_id='BP-MA-v0' \
                        --n_agent=2 \
                        --env_terminate_step=100 \
                        --big_box_reward=300 \
                        --a_lr=0.0003 \
                        --c_lr=0.003 \
                        --train_freq=32 \
                        --n_env=32 \
                        --c_target_update_freq=128 \
                        --n_step_TD=0 \
                        --grad_clip_norm=0 \
                        --eps_start=1.0 \
                        --eps_end=0.01 \
                        --eps_stable_at=4_000 \
                        --total_epi=40_000 \
                        --grid_dim 6 6 \
                        --gamma=0.98 \
                        --eval_policy \
                        --sample_epi \
                        --run_id=$i &
done

# Box Pushing 8x8
for ((i=0; i<20; i++))
do
    ma_niacc_rnn_sV.py --save_dir='ma_niacc_bp8' \
                        --l_mode=0 \
                        --env_id='BP-MA-v0' \
                        --n_agent=2 \
                        --env_terminate_step=100 \
                        --big_box_reward=300 \
                        --a_lr=0.0005 \
                        --c_lr=0.001 \
                        --train_freq=32 \
                        --n_env=32 \
                        --c_target_update_freq=128 \
                        --n_step_TD=0 \
                        --grad_clip_norm=0 \
                        --eps_start=1.0 \
                        --eps_end=0.01 \
                        --eps_stable_at=4_000 \
                        --total_epi=40_000 \
                        --grid_dim 8 8 \
                        --gamma=0.98 \
                        --eval_policy \
                        --sample_epi \
                        --run_id=$i &
done

# Box Pushing 10x10
for ((i=0; i<20; i++))
do
    ma_niacc_rnn_sV.py --save_dir='ma_niacc_bp10' \
                        --l_mode=0 \
                        --env_id='BP-MA-v0' \
                        --n_agent=2 \
                        --env_terminate_step=100 \
                        --big_box_reward=300 \
                        --a_lr=0.0005 \
                        --c_lr=0.001 \
                        --train_freq=32 \
                        --n_env=32 \
                        --c_target_update_freq=128 \
                        --n_step_TD=0 \
                        --grad_clip_norm=0 \
                        --eps_start=1.0 \
                        --eps_end=0.01 \
                        --eps_stable_at=6_000 \
                        --total_epi=40_000 \
                        --grid_dim 10 10 \
                        --gamma=0.98 \
                        --eval_policy \
                        --sample_epi \
                        --run_id=$i &
done

# Box Pushing 12x12
for ((i=0; i<20; i++))
do
    ma_niacc_rnn_sV.py --save_dir='ma_niacc_bp12' \
                        --l_mode=0 \
                        --env_id='BP-MA-v0' \
                        --n_agent=2 \
                        --env_terminate_step=100 \
                        --big_box_reward=300 \
                        --a_lr=0.0005 \
                        --c_lr=0.0005 \
                        --train_freq=48 \
                        --n_env=48 \
                        --c_target_update_freq=48 \
                        --n_step_TD=0 \
                        --grad_clip_norm=0 \
                        --eps_start=1.0 \
                        --eps_end=0.01 \
                        --eps_stable_at=6_000 \
                        --total_epi=40_000 \
                        --grid_dim 12 12 \
                        --gamma=0.98 \
                        --eval_policy \
                        --sample_epi \
                        --run_id=$i &
done

# Box Pushing 14x14
for ((i=0; i<20; i++))
do
    ma_niacc_rnn_sV.py --save_dir='ma_niacc_bp14' \
                        --l_mode=0 \
                        --env_id='BP-MA-v0' \
                        --n_agent=2 \
                        --env_terminate_step=100 \
                        --big_box_reward=300 \
                        --a_lr=0.0005 \
                        --c_lr=0.0005 \
                        --train_freq=32 \
                        --n_env=32 \
                        --c_target_update_freq=128 \
                        --n_step_TD=0 \
                        --grad_clip_norm=0 \
                        --eps_start=1.0 \
                        --eps_end=0.01 \
                        --eps_stable_at=8_000 \
                        --total_epi=40_000 \
                        --grid_dim 14 14 \
                        --gamma=0.98 \
                        --eval_policy \
                        --sample_epi \
                        --run_id=$i &
done
