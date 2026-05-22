from tests.jax_subprocess import OGBENCH_IMPLS, run_jax

_CHECK = """
    import utils.encoders as enc
    import utils.datasets as ds
    assert "drq" in enc.encoder_modules, "drq not registered"
    patched = getattr(ds.GCDataset.augment, "_ogbench_drq_patched", False)
    print("PATCHED", patched)
"""


def test_encoder_registered_and_augment_patched_with_flag():
    proc = run_jax(_CHECK, env_extra={"OGBENCH_DRQ_FRONTEND": "1"},
                   cwd=str(OGBENCH_IMPLS))
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "PATCHED True" in proc.stdout


def test_encoder_registered_but_augment_native_without_flag():
    proc = run_jax(_CHECK, cwd=str(OGBENCH_IMPLS))  # no flag
    assert proc.returncode == 0, f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    assert "PATCHED False" in proc.stdout  # drq registered, augment untouched
