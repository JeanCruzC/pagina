import importlib.util
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT)
spec = importlib.util.spec_from_file_location(
    "scheduler_mod", os.path.join(ROOT, "website", "scheduler.py")
)
scheduler_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler_mod)  # type: ignore
generate_shift_patterns = scheduler_mod.generate_shift_patterns
score_pattern = scheduler_mod.score_pattern


def test_generate_shift_patterns_top_k():
    # simple demand: two days with demand from 6 to 9
    dm = np.zeros((7, 24), dtype=int)
    dm[0, 6:10] = 1
    dm[1, 6:10] = 1
    cfg = {
        "use_ft": True,
        "use_pt": False,
        "allow_8h": True,
        "allow_10h8": False,
        "ACTIVE_DAYS": [0, 1],
    }
    # all patterns
    all_patterns = generate_shift_patterns(dm, top_k=20, cfg=cfg)
    scores = [s for s, _, _ in all_patterns]
    assert scores == sorted(scores, reverse=True)
    max_score = max(score_pattern(pat, dm) for _, _, pat in all_patterns)
    assert all_patterns[0][0] == max_score
    # top 3 should all have max_score and be drawn from the highest scoring set
    top3 = generate_shift_patterns(dm, top_k=3, cfg=cfg)
    assert len(top3) == 3
    assert all(s == max_score for s, _, _ in top3)
