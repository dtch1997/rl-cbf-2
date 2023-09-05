from absl import app
from absl import flags
from ml_collections import config_flags

from rl_cbf_2.algos.cql import *
from rl_cbf_2.algos.cql import _CONFIG

def eval(_):
    config = _CONFIG.value
    env = datasets.get_environment(config.env)
    _dataset = datasets.get_dataset(config.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    dataset = d4rl.qlearning_dataset(env, _dataset)

    # Construct safety-relabelled dataset
    safety_fn = datasets.get_safety_condition(config.env)
    dataset = datasets.relabel_dataset(dataset, config.relabel_type, safety_fn)

    if config.normalize_reward:
        modify_reward(
            dataset,
            config.env,
            reward_scale=config.reward_scale,
            reward_bias=config.reward_bias,
        )

    if config.normalize:
        state_mean, state_std = compute_mean_std(dataset["observations"], eps=1e-3)
    else:
        state_mean, state_std = 0, 1

    dataset["observations"] = normalize_states(
        dataset["observations"], state_mean, state_std
    )
    dataset["next_observations"] = normalize_states(
        dataset["next_observations"], state_mean, state_std
    )
    env = wrap_env(env, state_mean=state_mean, state_std=state_std)

    max_action = float(env.action_space.high[0])

    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    # Set seeds
    seed = config.seed
    set_seed(seed, env)

    critic_1 = FullyConnectedQFunction(
        state_dim,
        action_dim,
        config.orthogonal_init,
        config.q_n_hidden_layers,
    ).to(config.device)
    critic_2 = FullyConnectedQFunction(state_dim, action_dim, config.orthogonal_init).to(
        config.device
    )
    critic_1_optimizer = torch.optim.Adam(list(critic_1.parameters()), config.qf_lr)
    critic_2_optimizer = torch.optim.Adam(list(critic_2.parameters()), config.qf_lr)

    actor = TanhGaussianPolicy(
        state_dim,
        action_dim,
        max_action,
        log_std_multiplier=config.policy_log_std_multiplier,
        orthogonal_init=config.orthogonal_init,
    ).to(config.device)
    actor_optimizer = torch.optim.Adam(actor.parameters(), config.policy_lr)

    kwargs = {
        "critic_1": critic_1,
        "critic_2": critic_2,
        "critic_1_optimizer": critic_1_optimizer,
        "critic_2_optimizer": critic_2_optimizer,
        "actor": actor,
        "actor_optimizer": actor_optimizer,
        "discount": config.discount,
        "soft_target_update_rate": config.soft_target_update_rate,
        "device": config.device,
        # CQL
        "target_entropy": -np.prod(env.action_space.shape).item(),
        "alpha_multiplier": config.alpha_multiplier,
        "use_automatic_entropy_tuning": config.use_automatic_entropy_tuning,
        "backup_entropy": config.backup_entropy,
        "policy_lr": config.policy_lr,
        "qf_lr": config.qf_lr,
        "bc_steps": config.bc_steps,
        "target_update_period": config.target_update_period,
        "cql_n_actions": config.cql_n_actions,
        "cql_importance_sample": config.cql_importance_sample,
        "cql_lagrange": config.cql_lagrange,
        "cql_target_action_gap": config.cql_target_action_gap,
        "cql_temp": config.cql_temp,
        "cql_alpha": config.cql_alpha,
        "cql_max_target_backup": config.cql_max_target_backup,
        "cql_clip_diff_min": config.cql_clip_diff_min,
        "cql_clip_diff_max": config.cql_clip_diff_max,
    }

    # Initialize actor
    trainer = ContinuousCQL(**kwargs)

    if config.load_model != "":
        policy_file = Path(config.load_model)
        trainer.load_state_dict(torch.load(policy_file))
        actor = trainer.actor

    if config.dry_run:
        print("Dry run, exiting")
        return

    # Evaluate episode
    eval_scores = eval_actor(
        env,
        actor,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    eval_score = eval_scores.mean()
    eval_log = {}

    normalized_eval_score = datasets.get_normalized_score(config.env, eval_score) * 100.0
    eval_log["eval/d4rl_normalized_score"] = normalized_eval_score
    print("---------------------------------------")
    print(
        f"Evaluation over {config.n_episodes} episodes: "
        f"{eval_score:.3f} , D4RL score: {normalized_eval_score:.3f}"
    )
    print("---------------------------------------")

    # Evaluate CBF
    eval_metrics = eval_cbf(
        env, 
        actor, 
        critic_1,
        critic_2,
        device=config.device,
        n_episodes=config.n_episodes,
        seed=config.seed,
    )
    for k, v in eval_metrics.items():
        eval_log[f"eval_cbf/{k}"] = v
    print(eval_log)


if __name__ == "__main__":
    app.run(eval)