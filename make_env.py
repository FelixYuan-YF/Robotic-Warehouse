from warehouse import RewardType, Warehouse

def load_env(shelf_columns=3, column_height=8, shelf_rows=1, n_agents=2, msg_bits=0, 
             sensor_range=1, request_queue_size=2, max_inactivity_steps=None, 
             max_steps=500, reward_type=RewardType.INDIVIDUAL, render_mode='human'):
    env = Warehouse(
        shelf_columns=shelf_columns,
        column_height=column_height,
        shelf_rows=shelf_rows,
        n_agents=n_agents,
        msg_bits=msg_bits,
        sensor_range=sensor_range,
        request_queue_size=request_queue_size,
        max_inactivity_steps=max_inactivity_steps,
        max_steps=max_steps,
        reward_type=reward_type,
        render_mode=render_mode
    )
    num_agents = env.n_agents
    state_size = env.observation_space[0].shape[0]
    action_size = env.action_space[0].n
    return env, num_agents, state_size, action_size