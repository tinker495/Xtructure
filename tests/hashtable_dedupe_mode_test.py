import os
import subprocess
import sys
import textwrap
from pathlib import Path


def test_parallel_insert_approx_mode_smoke():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["XTRUCTURE_HASHTABLE_DEDUPE_MODE"] = "approx"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), env.get("PYTHONPATH", "")] if env.get("PYTHONPATH") else [str(repo_root)]
    )

    code = textwrap.dedent(
        """
        import jax
        import jax.numpy as jnp

        from xtructure import FieldDescriptor, HashTable, xtructure_dataclass


        @xtructure_dataclass(validate=False)
        class InsertApproxValue:
            a: FieldDescriptor.scalar(dtype=jnp.uint32, default=0)
            b: FieldDescriptor.tensor(dtype=jnp.uint16, shape=(8,), fill_value=0)


        batch = 128
        key = jax.random.PRNGKey(0)
        values = InsertApproxValue.random((batch,), key=key)

        table = HashTable.build(InsertApproxValue, 1, 512)
        table, inserted, unique, idx = table.parallel_insert(values)
        idx2, found = table.lookup_parallel(values)

        assert found.shape == (batch,)
        assert bool(jax.device_get(jnp.all(found)))
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            "Subprocess failed with XTRUCTURE_HASHTABLE_DEDUPE_MODE=approx\n"
            f"stdout:\n{proc.stdout}\n\n"
            f"stderr:\n{proc.stderr}\n"
        )
