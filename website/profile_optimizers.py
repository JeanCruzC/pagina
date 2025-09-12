# -*- coding: utf-8 -*-
"""
Optimizadores de perfiles específicos
"""

import random
import time
from functools import wraps

def single_model(func):
    """Decorador para exclusión mutua de modelo"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

def merge_config(cfg):
    """Mezcla configuración base"""
    from .scheduler_core import merge_config as _merge_config
    return _merge_config(cfg)

def _score_result(assignments, shifts_coverage, demand_matrix, target=100.0, coverage_method="efficiency", penalty_factor=1.0):
    """Score unificado usando la métrica única de cobertura.

    Nota: coverage_method y penalty_factor se ignoran; se usa siempre la fórmula simétrica.
    """
    from .scheduler import analyze_results
    res = analyze_results(assignments, shifts_coverage, demand_matrix)
    if not res:
        return float("inf"), {}
    
    cov = res.get("coverage_percentage", 0.0)
    over = float(res.get("overstaffing", 0.0))
    under = float(res.get("understaffing", 0.0))
    total_dem = float(demand_matrix.sum() or 1.0)
    penalty_balance = (over + under) / total_dem * 100.0
    return abs(target - cov) + 0.25 * penalty_balance, res

def _sample_space():
    """Espacio pequeño y estable (rápido). Ajusta rangos si lo necesitas."""
    return {
        "agent_limit_factor": random.randint(8, 28),
        "excess_penalty": 10 ** random.uniform(-2, 0.7),  # ~[0.01, ~5]
        "peak_bonus": random.uniform(1.0, 3.5),
        "critical_bonus": random.uniform(1.0, 3.5),
        "allow_deficit": False,
        "allow_excess": True,   # dejamos un poco de exceso en HPO para evitar estancarse
    }

def _try_import_optuna():
    try:
        import optuna
        return optuna
    except Exception:
        return None

def _hpo_unico(shifts_coverage, demand_matrix, base_cfg, n_trials=12, job_id=None):
    """HPO rápido común a todos los solvers (eval barata con greedy_fast / chunks)."""
    cfg = merge_config(base_cfg)
    optuna = _try_import_optuna()

    def eval_cfg(params):
        cand = cfg.copy()
        cand.update(params)
        try:
            from .optimizer_greedy import optimize_greedy_fast
            assign, _ = optimize_greedy_fast(shifts_coverage, demand_matrix, cfg=cand, job_id=job_id)
        except Exception:
            from .scheduler import solve_in_chunks_optimized
            assign = solve_in_chunks_optimized(shifts_coverage, demand_matrix, **cand)
        score, _ = _score_result(assign, shifts_coverage, demand_matrix, target=cand.get("TARGET_COVERAGE", 100.0))
        return score, cand

    # Ruta A: Optuna si está disponible (TPE)
    if optuna:
        def objective(trial):
            params = {
                "agent_limit_factor": trial.suggest_int("agent_limit_factor", 8, 28),
                "excess_penalty": trial.suggest_float("excess_penalty", 0.01, 5.0, log=True),
                "peak_bonus": trial.suggest_float("peak_bonus", 1.0, 3.5),
                "critical_bonus": trial.suggest_float("critical_bonus", 1.0, 3.5),
                "allow_excess": True,
                "allow_deficit": False,
            }
            score, _ = eval_cfg(params)
            return score

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        best.update({"allow_excess": True, "allow_deficit": False})
        return best

    # Ruta B: Random Search (sin dependencias)
    best_params, best_score = None, float("inf")
    for _ in range(max(3, n_trials)):
        params = _sample_space()
        score, _ = eval_cfg(params)
        if score < best_score:
            best_params, best_score = params, score
            if best_score < 1e-3:  # prácticamente perfecto
                break
    return best_params or {}

@single_model
def optimize_hpo_then_solve(shifts_coverage, demand_matrix, *, cfg=None, job_id=None):
    """
    1) Busca hiperparámetros una sola vez (rápido).
    2) Ejecuta cascada con esa config: PuLP -> Greedy -> Chunks.
    3) Devuelve la mejor asignación según el score unificado.
    """
    cfg = merge_config(cfg)
    target = cfg.get("TARGET_COVERAGE", 100.0)
    trials = int(cfg.get("hpo_trials", 12))

    # Fase 1: HPO único
    best_cfg = _hpo_unico(shifts_coverage, demand_matrix, cfg, n_trials=trials, job_id=job_id)
    run_cfg = cfg.copy()
    run_cfg.update(best_cfg)

    # Fase 2: Cascada
    candidates = []

    # 2a) PuLP
    try:
        from .optimizer_pulp import optimize_with_pulp
        a_pulp, s_pulp = optimize_with_pulp(
            shifts_coverage, demand_matrix, cfg=run_cfg, job_id=job_id
        )
        if a_pulp:
            sc, m = _score_result(a_pulp, shifts_coverage, demand_matrix, target=target)
            candidates.append(("PULP", sc, a_pulp, m))
    except Exception:
        pass

    # 2b) Greedy completo
    try:
        from .optimizer_greedy import optimize_with_greedy
        a_greedy, s_gr = optimize_with_greedy(shifts_coverage, demand_matrix, cfg=run_cfg, job_id=job_id)
        if a_greedy:
            sc, m = _score_result(a_greedy, shifts_coverage, demand_matrix, target=target)
            candidates.append(("GREEDY", sc, a_greedy, m))
    except Exception:
        pass

    # 2c) Fallback: chunks
    try:
        from .scheduler import solve_in_chunks_optimized
        a_chunks = solve_in_chunks_optimized(shifts_coverage, demand_matrix, **run_cfg)
        if a_chunks:
            sc, m = _score_result(a_chunks, shifts_coverage, demand_matrix, target=target)
            candidates.append(("CHUNKS", sc, a_chunks, m))
    except Exception:
        pass

    if not candidates:
        return {}, "HPO_CASCADE_NO_SOLUTION"

    # Elige la mejor por menor score
    candidates.sort(key=lambda t: t[1])
    best_name, best_score, best_assign, best_metrics = candidates[0]
    print(f"[HPO+CASCADA] Mejor: {best_name} | score={best_score:.3f} | cov={best_metrics.get('coverage_percentage', 0):.1f}%")
    return best_assign, f"HPO_CASCADE_{best_name}"

# Registro de perfiles
PROFILE_OPTIMIZERS = {
    "HPO + Cascada 100%": optimize_hpo_then_solve,
    "HPO y Cascada": optimize_hpo_then_solve,  # alias opcional
}

def get_profile_optimizer(profile_name: str):
    """Devuelve el optimizador para el perfil dado"""
    return PROFILE_OPTIMIZERS.get(profile_name)
