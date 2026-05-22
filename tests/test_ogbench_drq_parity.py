import os
import subprocess
import tempfile
import textwrap
from pathlib import Path

import numpy as np

from tests.jax_subprocess import OGBENCH_IMPLS, SHIM_DIR, jax_python


def test_drq_jax_matches_fb_torch_under_transplant():
    import gymnasium
    import torch
    from nn_models import DrQEncoderArchiConfig

    torch.manual_seed(0)
    obs_space = gymnasium.spaces.Box(0, 255, (9, 64, 64), np.uint8)  # CHW, obs-alone
    enc = DrQEncoderArchiConfig(feature_dim=256).build(obs_space).eval()

    x_raw = torch.rand(2, 9, 64, 64) * 255.0          # [0,255] float, NCHW
    with torch.no_grad():
        out_t = enc(x_raw / 255.0 - 0.5).numpy()      # FB applies RGBNorm outside
    sd = enc.state_dict()
    # trunk = Sequential(Conv,ReLU,Conv,ReLU,Conv,ReLU,Conv,ReLU,Flatten) -> convs at 0,2,4,6
    # proj  = Sequential(Linear, LayerNorm, Tanh) -> Linear=proj.0, LayerNorm=proj.1
    arrs = dict(
        conv0_w=sd["trunk.0.weight"].numpy(), conv0_b=sd["trunk.0.bias"].numpy(),
        conv1_w=sd["trunk.2.weight"].numpy(), conv1_b=sd["trunk.2.bias"].numpy(),
        conv2_w=sd["trunk.4.weight"].numpy(), conv2_b=sd["trunk.4.bias"].numpy(),
        conv3_w=sd["trunk.6.weight"].numpy(), conv3_b=sd["trunk.6.bias"].numpy(),
        proj_w=sd["proj.0.weight"].numpy(),   proj_b=sd["proj.0.bias"].numpy(),
        ln_w=sd["proj.1.weight"].numpy(),     ln_b=sd["proj.1.bias"].numpy(),
        x_raw=x_raw.numpy(), out_t=out_t,
    )
    with tempfile.TemporaryDirectory() as d:
        npz = Path(d) / "drq_parity.npz"
        np.savez(npz, **arrs)

        jax_code = textwrap.dedent(
            """
            import sys, numpy as np, jax, jax.numpy as jnp
            from ogbench_drq import DrQEncoder
            d = np.load(sys.argv[1])
            x = jnp.asarray(d["x_raw"]).transpose(0, 2, 3, 1)        # NCHW -> NHWC
            enc = DrQEncoder(feature_dim=256)
            enc.init(jax.random.PRNGKey(0), x)                       # shape check only
            def ck(w):  # torch [out,in,kh,kw] -> flax [kh,kw,in,out]
                return jnp.asarray(w).transpose(2, 3, 1, 0)
            # proj: torch flatten order (C,H,W); flax reshape order (H,W,C).
            # reorder torch Linear weight columns C,H,W -> H,W,C, then to [in,out].
            pw = d["proj_w"].reshape(256, 32, 25, 25)                # [out,C,H,W]
            pw = np.transpose(pw, (2, 3, 1, 0)).reshape(20000, 256)  # [H*W*C, out]
            params = {
                "conv0": {"kernel": ck(d["conv0_w"]), "bias": jnp.asarray(d["conv0_b"])},
                "conv1": {"kernel": ck(d["conv1_w"]), "bias": jnp.asarray(d["conv1_b"])},
                "conv2": {"kernel": ck(d["conv2_w"]), "bias": jnp.asarray(d["conv2_b"])},
                "conv3": {"kernel": ck(d["conv3_w"]), "bias": jnp.asarray(d["conv3_b"])},
                "proj": {"kernel": jnp.asarray(pw), "bias": jnp.asarray(d["proj_b"])},
                "proj_ln": {"scale": jnp.asarray(d["ln_w"]), "bias": jnp.asarray(d["ln_b"])},
            }
            out_j = np.asarray(enc.apply({"params": params}, x))
            diff = float(np.max(np.abs(out_j - d["out_t"])))
            print("MAXDIFF", diff)
            sys.exit(0 if diff < 1e-5 else 1)
            """
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(SHIM_DIR), str(OGBENCH_IMPLS), env.get("PYTHONPATH", "")]
        )
        env["JAX_PLATFORMS"] = "cpu"
        proc = subprocess.run(
            [jax_python(), "-c", jax_code, str(npz)],
            env=env, capture_output=True, text=True,
        )
    assert proc.returncode == 0, (
        f"parity failed\nstdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    assert "MAXDIFF" in proc.stdout
