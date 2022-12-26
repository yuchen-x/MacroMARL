#/bin/bash
# Overcooked-A
for ((i=0; i<20; i++))
do
     value_based_main.py  --save_dir='ma_cen_condi_ddrqn_overcooked_A' \
                                --alg='MacCenQ' \
                                --env_id='Overcooked-MA-v1' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --batch_size=64 \
                                --train_freq=64 \
                                --total_epi=100_000 \
                                --replay_buffer_size=1000 \
                                --l_rate=0.00003 \
                                --h_stable_at=20_000 \
                                --eps_l_d_steps=20_000 \
                                --eps_end=0.05 \
                                --target_update_freq=5000 \
                                --discount=0.99 \
                                --start_train=1000 \
                                --mlp_layer_size 128 64 \
                                --rnn_h_size=64 \
                                --grid_dim 7 7 \
                                --task=6 \
                                --map_type=A \
                                --step_penalty=-0.1 \
                                --sample_epi \
                                --h_explore \
                                --init_h 1.0 \
                                --end_h 1.0 \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done

# Overcooked-B
for ((i=0; i<20; i++))
do
    value_based_main.py  --save_dir='ma_cen_condi_ddrqn_overcooked_B' \
                                --alg='MacCenQ' \
                                --env_id='Overcooked-MA-v1' \
                                --n_agent=3 \
                                --env_terminate_step=200 \
                                --batch_size=64 \
                                --train_freq=64 \
                                --total_epi=100_000 \
                                --replay_buffer_size=500 \
                                --l_rate=0.0001 \
                                --h_stable_at=20_000 \
                                --eps_l_d_steps=20_000 \
                                --eps_end=0.05 \
                                --target_update_freq=5000 \
                                --discount=0.99 \
                                --start_train=500 \
                                --mlp_layer_size 128 64 \
                                --rnn_h_size=64 \
                                --grid_dim 7 7 \
                                --task=6 \
                                --map_type=B \
                                --step_penalty=-0.1 \
                                --sample_epi \
                                --h_explore \
                                --init_h 1.0 \
                                --end_h 1.0 \
                                --dynamic_h \
                                --eps_l_d \
                                --run_id=$i & 
done
