#!/usr/bin/env python3
"""Independent FP64 reconstructed K/V oracle (no projected-query factoring)."""
import json
import sys
from pathlib import Path
import numpy as np


def validate(path):
    d = json.loads(Path(path).read_text())
    q = np.array(d['q']).reshape(16, 64)
    z = np.array(d['latent_logical_prequantization'], dtype=np.float32).astype(np.float16).astype(np.float64).reshape(-1, 32)
    kp = np.array(d['kp']).reshape(32, 4, 64)
    vp = np.array(d['vp']).reshape(32, 4, 64)
    n = len(z)
    table = np.array(d['table'])
    assert np.array_equal(table, (17*np.arange(n//16)+11) % (n//16))
    assert len(set(table)) == n//16 and not np.array_equal(table, np.arange(n//16))
    # z is logical; GPU must gather the permuted physical cache to match this.
    k = np.einsum('tl,lhd->thd', z, kp)
    v = np.einsum('tl,lhd->thd', z, vp)
    scores = np.einsum('hd,thd->ht', q, k[:, np.arange(16)//4, :]) / 8
    probs = np.exp(scores-scores.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    context = np.einsum('ht,thd->hd', probs, v[:, np.arange(16)//4, :])
    result = {'seq': n, 'oracle': 'FP64 reconstructed K/V from FP16 latent; FP32 input q/projections', 'atol': 0.005, 'rtol': 0.0, 'relative_floor': 1e-12, 'pass': True}
    for name, ref in [('scores', scores), ('probabilities', probs), ('context', context)]:
        actual = np.array(d[name]).reshape(ref.shape)
        errors = np.abs(actual-ref)
        failures = np.argwhere(~np.isfinite(actual) | (errors > result['atol']))
        result[name] = {'max_absolute_error': float(errors.max()), 'max_relative_error': float((errors/np.maximum(np.abs(ref), 1e-12)).max()), 'max_absolute_location': list(map(int,np.unravel_index(errors.argmax(),ref.shape))), 'first_failure': failures[0].tolist() if len(failures) else None}
        result['pass'] &= len(failures) == 0
    result['normalization_error'] = float(np.abs(np.array(d['probabilities']).reshape(16,n).sum(1)-1).max())
    result['pass'] &= result['normalization_error'] <= 1e-4
    return result

if __name__ == '__main__':
    result = validate(sys.argv[1])
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['pass'] else 1)
