def test_short_train_and_checkpoint(tmp_path):
    import jax.numpy as jnp

    from impls.agent import Agent
    from impls.utils.config_utils import load_config
    from impls.utils.env_utils import make_env_and_datasets
    from impls.utils.flax_utils import restore_agent, save_agent

    env, train_ds, _ = make_env_and_datasets("cube-single-play-v0")
    cfg = load_config("fb", "flowbc", "cube_single")
    ex = train_ds.sample(2)
    agent = Agent.create(
        "fb",
        "flowbc",
        0,
        jnp.asarray(ex["observations"]),
        jnp.asarray(ex["actions"]),
        cfg,
    )

    infos = []
    for _ in range(10):
        batch = train_ds.sample(cfg["batch_size"])
        agent, info = agent.update(batch)
        infos.append(info)
    assert all(jnp.isfinite(jnp.asarray(i["fb_loss"])).all() for i in infos)
    assert all(jnp.isfinite(jnp.asarray(i["actor_loss"])).all() for i in infos)

    save_agent(agent, str(tmp_path), 10)
    restored = restore_agent(agent, str(tmp_path), 10)
    assert restored.net["forward"].step == agent.net["forward"].step
