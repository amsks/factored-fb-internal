# tests/test_envs.py
def test_domain_lists():
    from envs.ogbench import ANT_DOMAINS, CUBE_DOMAINS, ALL_DOMAINS, ALL_TASKS
    assert "antmaze-medium-navigate-v0" in ANT_DOMAINS
    assert "cube-single-play-v0" in CUBE_DOMAINS
    assert len(ALL_TASKS) == len(ALL_DOMAINS)
    # Each domain has 5 tasks
    for domain, tasks in ALL_TASKS.items():
        assert len(tasks) == 5, f"{domain} should have 5 tasks, got {len(tasks)}"

def test_rollout_import():
    from envs.rollout import rollout
    assert callable(rollout)

def test_pixel_wrapper_import():
    from envs.wrappers import PixelWrapper
    assert PixelWrapper is not None
