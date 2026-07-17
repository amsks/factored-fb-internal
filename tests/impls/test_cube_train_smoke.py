def test_short_train_and_checkpoint(tmp_path):
    import jax.numpy as jnp
    from impls.agents import Agent
    from impls.utils.env_utils import make_env_and_datasets
    from impls.utils.flax_utils import save_agent, restore_agent

    env, train_ds, _ = make_env_and_datasets("cube-single-play-v0")
    from impls.main import get_config
    cfg = get_config()
    ex = train_ds.sample(2)
    agent = Agent.create("fb", "flowbc", 0, jnp.asarray(ex["observations"]), jnp.asarray(ex["actions"]), cfg)

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
