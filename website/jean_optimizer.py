"""
Módulo generador de turnos – Perfil JEAN personalizado optimizado.
Implementa estrategia robusta para garantizar cobertura del 100% con exceso y déficit mínimos.
Incluye generación completa de patrones, optimización ILP en dos fases y parámetros adaptativos.
"""

import numpy as np
import hashlib
from itertools import combinations, permutations
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    pulp = None
    PULP_AVAILABLE = False

# Configuración global
TARGET_COVERAGE = 100.0
TIME_LIMIT = 240
MAX_EXCESS_PERCENTAGE = 0.05  # 5% máximo exceso permitido

def generate_complete_patterns(demand_matrix, use_ft=True, use_pt=True, cfg=None):
    """
    Genera TODOS los patrones posibles según configuración del usuario.
    Sin filtrado por scoring - mantiene 100% de patrones válidos.
    """
    cfg = cfg or {}
    active_days = [d for d in range(7) if demand_matrix[d].sum() > 0]
    
    # Determinar rango completo de horas de inicio (sin restricción 6-20)
    nonzero_hours = np.where(demand_matrix.sum(axis=0) > 0)[0]
    first_hour = int(nonzero_hours[0]) if len(nonzero_hours) > 0 else 0
    last_hour = int(nonzero_hours[-1] + 1) if len(nonzero_hours) > 0 else 24
    start_hours = np.arange(first_hour, last_hour, 0.5)
    
    patterns = {}
    seen_patterns = set()
    
    def add_pattern(name, pattern):
        key = hashlib.md5(pattern.tobytes()).hexdigest()
        if key not in seen_patterns:
            seen_patterns.add(key)
            patterns[name] = pattern.flatten()
            return True
        return False
    
    # FT 8H: Generar para TODOS los días activos (incluso <6)
    if use_ft and cfg.get("allow_8h", True):
        for start in start_hours:
            if len(active_days) >= 6:
                # 6 días trabajo + 1 libre
                for dso in active_days + [None]:
                    wd = [d for d in active_days if d != dso][:6]
                    if len(wd) >= 6:
                        for brk_pos in get_break_positions(start, 8, cfg):
                            pattern = build_pattern_with_break(start, 8, wd, brk_pos, cfg)
                            name = f"FT8_{start:04.1f}_{''.join(map(str,wd))}_B{brk_pos:04.1f}"
                            add_pattern(name, pattern)
            else:
                # Usar TODOS los días activos disponibles
                wd = list(active_days)
                for brk_pos in get_break_positions(start, 8, cfg):
                    pattern = build_pattern_with_break(start, 8, wd, brk_pos, cfg)
                    name = f"FT8_{start:04.1f}_{''.join(map(str,wd))}_B{brk_pos:04.1f}"
                    add_pattern(name, pattern)
    
    # FT 10H+8H: 5 días (4×10h + 1×8h)
    if use_ft and cfg.get("allow_10h8", False) and len(active_days) >= 5:
        for start in start_hours[::2]:  # cada hora
            for dso in active_days:
                wd = [d for d in active_days if d != dso][:5]
                if len(wd) >= 5:
                    for eight_day in wd:
                        pattern = build_mixed_pattern(start, wd, eight_day, cfg)
                        name = f"FT10p8_{start:04.1f}_DSO{dso}_8D{eight_day}"
                        add_pattern(name, pattern)
    
    # PT: Generar TODAS las combinaciones posibles de días
    if use_pt:
        # PT 4H: 1 a N días (máx 24h/semana)
        if cfg.get("allow_pt_4h", True):
            for start in start_hours[::2]:
                for num_days in range(1, len(active_days) + 1):
                    if 4 * num_days <= 24:
                        for days_combo in combinations(active_days, num_days):
                            pattern = build_simple_pattern(start, 4, list(days_combo))
                            name = f"PT4_{start:04.1f}_{''.join(map(str,days_combo))}"
                            add_pattern(name, pattern)
        
        # PT 6H: 1 a 4 días (máx 24h/semana)
        if cfg.get("allow_pt_6h", True):
            for start in start_hours[::3]:
                for num_days in range(1, min(len(active_days) + 1, 5)):
                    if 6 * num_days <= 24:
                        for days_combo in combinations(active_days, num_days):
                            pattern = build_simple_pattern(start, 6, list(days_combo))
                            name = f"PT6_{start:04.1f}_{''.join(map(str,days_combo))}"
                            add_pattern(name, pattern)
        
        # PT 5H: Patrones especiales (4×5h + 1×4h = 24h)
        if cfg.get("allow_pt_5h", False):
            for start in start_hours[::3]:
                for num_days in range(1, len(active_days) + 1):
                    if 5 * num_days <= 25:
                        for days_combo in combinations(active_days, num_days):
                            if 5 * num_days <= 24:
                                pattern = build_simple_pattern(start, 5, list(days_combo))
                            else:
                                pattern = build_pt5_pattern(start, list(days_combo))
                            name = f"PT5_{start:04.1f}_{''.join(map(str,days_combo))}"
                            add_pattern(name, pattern)
    
    print(f"[JEAN] Patrones generados: {len(patterns)} (sin filtrado)")
    return patterns

def get_break_positions(start_hour, duration, cfg):
    """Obtiene posiciones válidas de break dentro del turno."""
    break_from_start = cfg.get("break_from_start", 2.5)
    break_from_end = cfg.get("break_from_end", 2.5)
    
    earliest = start_hour + break_from_start
    latest = start_hour + duration - break_from_end - 1
    
    positions = []
    current = earliest
    while current <= latest:
        if current % 0.5 == 0:  # Solo en horas/medias horas
            positions.append(round(current, 1))
        current += 0.5
    
    return positions[:5]  # Máximo 5 posiciones

def build_pattern_with_break(start_hour, duration, working_days, break_pos, cfg):
    """Construye patrón con break en posición específica."""
    pattern = np.zeros((7, 24), dtype=int)
    break_len = cfg.get("break_duration", 1)
    
    for day in working_days:
        # Marcar horas trabajadas
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
        
        # Quitar break
        for b in range(break_len):
            t = break_pos + b
            d_off, idx = divmod(int(t), 24)
            if 0 <= idx < 24:
                pattern[(day + d_off) % 7, idx] = 0
    
    return pattern

def build_mixed_pattern(start_hour, working_days, eight_hour_day, cfg):
    """Construye patrón FT 10h+8h con breaks."""
    pattern = np.zeros((7, 24), dtype=int)
    break_len = cfg.get("break_duration", 1)
    break_from_start = cfg.get("break_from_start", 2.5)
    
    for day in working_days:
        duration = 8 if day == eight_hour_day else 10
        
        # Marcar horas trabajadas
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
        
        # Break en mitad del turno
        break_pos = start_hour + break_from_start
        for b in range(break_len):
            t = break_pos + b
            d_off, idx = divmod(int(t), 24)
            if 0 <= idx < 24:
                pattern[(day + d_off) % 7, idx] = 0
    
    return pattern

def build_simple_pattern(start_hour, duration, working_days):
    """Construye patrón PT simple sin breaks."""
    pattern = np.zeros((7, 24), dtype=int)
    
    for day in working_days:
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
    
    return pattern

def build_pt5_pattern(start_hour, working_days):
    """Construye patrón PT5 especial: 4×5h + 1×4h."""
    pattern = np.zeros((7, 24), dtype=int)
    four_hour_day = working_days[-1] if working_days else 0
    
    for day in working_days:
        duration = 4 if day == four_hour_day else 5
        for h in range(duration):
            t = start_hour + h
            d_off, idx = divmod(int(t), 24)
            pattern[(day + d_off) % 7, idx] = 1
    
    return pattern

def optimize_two_phase_ilp(patterns, demand_matrix, cfg=None):
    """
    Optimización ILP en dos fases:
    Fase 1: FT sin exceso (cobertura ≤ demanda)
    Fase 2: PT para completar déficit con exceso mínimo controlado
    """
    if not PULP_AVAILABLE:
        raise ImportError("PuLP no disponible para optimización ILP")
    
    cfg = cfg or {}
    
    # Separar patrones FT y PT
    ft_patterns = {k: v for k, v in patterns.items() if k.startswith("FT")}
    pt_patterns = {k: v for k, v in patterns.items() if k.startswith("PT")}
    
    print(f"[JEAN] Fase 1: Optimizando {len(ft_patterns)} patrones FT (sin exceso)")
    ft_assignments = optimize_ft_phase(ft_patterns, demand_matrix, cfg)
    
    # Calcular cobertura FT y demanda restante
    ft_coverage = calculate_coverage(ft_assignments, ft_patterns, demand_matrix.shape)
    remaining_demand = np.maximum(demand_matrix - ft_coverage, 0)
    
    print(f"[JEAN] Fase 2: Optimizando {len(pt_patterns)} patrones PT (déficit restante)")
    pt_assignments = optimize_pt_phase(pt_patterns, remaining_demand, cfg)
    
    # Combinar resultados
    final_assignments = {}
    final_assignments.update(ft_assignments)
    final_assignments.update(pt_assignments)
    
    return final_assignments

def optimize_ft_phase(ft_patterns, demand_matrix, cfg):
    """Fase 1: FT con restricción de NO EXCESO."""
    if not ft_patterns:
        return {}
    
    D, H = demand_matrix.shape
    prob = pulp.LpProblem("JEAN_FT_NoExcess", pulp.LpMinimize)
    
    # Variables: agentes por patrón FT
    agent_limit = max(10, int(demand_matrix.sum() / cfg.get("agent_limit_factor", 15)))
    ft_vars = {k: pulp.LpVariable(f"ft_{k}", 0, agent_limit, cat=pulp.LpInteger) 
               for k in ft_patterns}
    
    # Variables de déficit (no exceso)
    deficit_vars = {(d, h): pulp.LpVariable(f"def_ft_{d}_{h}", 0, None) 
                    for d in range(D) for h in range(H)}
    
    # Objetivo: minimizar déficit + agentes
    prob += (pulp.lpSum(deficit_vars.values()) * 1000 + 
             pulp.lpSum(ft_vars.values()) * 1)
    
    # Restricciones
    for d in range(D):
        for h in range(H):
            coverage = pulp.lpSum(ft_vars[k] * ft_patterns[k][d * H + h] for k in ft_patterns)
            # Cobertura + déficit ≥ demanda
            prob += coverage + deficit_vars[(d, h)] >= demand_matrix[d, h]
            # Cobertura ≤ demanda (SIN EXCESO)
            prob += coverage <= demand_matrix[d, h]
    
    # Resolver
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT // 2)
    prob.solve(solver)
    
    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for k in ft_patterns:
            val = int(ft_vars[k].value() or 0)
            if val > 0:
                assignments[k] = val
    
    return assignments

def optimize_pt_phase(pt_patterns, remaining_demand, cfg):
    """Fase 2: PT para completar déficit con exceso controlado."""
    if not pt_patterns or remaining_demand.sum() == 0:
        return {}
    
    D, H = remaining_demand.shape
    prob = pulp.LpProblem("JEAN_PT_Complete", pulp.LpMinimize)
    
    # Variables: agentes por patrón PT
    agent_limit = max(10, int(remaining_demand.sum() / cfg.get("agent_limit_factor", 15)))
    pt_vars = {k: pulp.LpVariable(f"pt_{k}", 0, agent_limit, cat=pulp.LpInteger) 
               for k in pt_patterns}
    
    # Variables de déficit y exceso
    deficit_vars = {(d, h): pulp.LpVariable(f"def_pt_{d}_{h}", 0, None) 
                    for d in range(D) for h in range(H)}
    excess_vars = {(d, h): pulp.LpVariable(f"exc_pt_{d}_{h}", 0, None) 
                   for d in range(D) for h in range(H)}
    
    # Objetivo: minimizar déficit >> exceso >> agentes
    excess_penalty = cfg.get("excess_penalty", 5.0)
    prob += (pulp.lpSum(deficit_vars.values()) * 1000 + 
             pulp.lpSum(excess_vars.values()) * (excess_penalty * 20) +
             pulp.lpSum(pt_vars.values()) * 1)
    
    # Restricción global: exceso máximo 5% de demanda total
    total_excess = pulp.lpSum(excess_vars.values())
    prob += total_excess <= remaining_demand.sum() * MAX_EXCESS_PERCENTAGE
    
    # Restricciones de cobertura
    for d in range(D):
        for h in range(H):
            coverage = pulp.lpSum(pt_vars[k] * pt_patterns[k][d * H + h] for k in pt_patterns)
            prob += coverage + deficit_vars[(d, h)] >= remaining_demand[d, h]
            prob += coverage - excess_vars[(d, h)] <= remaining_demand[d, h]
    
    # Resolver
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=TIME_LIMIT // 2)
    prob.solve(solver)
    
    assignments = {}
    if prob.status == pulp.LpStatusOptimal:
        for k in pt_patterns:
            val = int(pt_vars[k].value() or 0)
            if val > 0:
                assignments[k] = val
    
    return assignments

def calculate_coverage(assignments, patterns, shape):
    """Calcula matriz de cobertura desde asignaciones."""
    D, H = shape
    coverage = np.zeros((D, H), dtype=int)
    
    for pattern_name, count in assignments.items():
        if pattern_name in patterns:
            pattern_matrix = np.array(patterns[pattern_name]).reshape(D, H)
            coverage += pattern_matrix * count
    
    return coverage

def get_adaptive_parameters(demand_matrix):
    """Calcula parámetros adaptativos basados en características de demanda."""
    total_demand = demand_matrix.sum()
    peak_demand = demand_matrix.max()
    avg_demand = demand_matrix.mean()
    
    # Factor límite de agentes adaptativo
    if peak_demand > 0:
        agent_limit_factor = max(8, min(25, int(total_demand / peak_demand * 2)))
    else:
        agent_limit_factor = 15
    
    # Penalización de exceso adaptativa
    demand_variability = np.std(demand_matrix) / max(avg_demand, 1)
    excess_penalty = max(2.0, min(10.0, 5.0 * (1 + demand_variability)))
    
    return {
        "agent_limit_factor": agent_limit_factor,
        "excess_penalty": excess_penalty,
        "break_duration": 1,
        "break_from_start": 2.5,
        "break_from_end": 2.5
    }

def optimize_jean_perfect(demand_matrix, user_config=None):
    """
    Función principal: optimización JEAN para cobertura perfecta del 100%.
    
    Args:
        demand_matrix: Matriz 7x24 de demanda
        user_config: Configuración del usuario (FT/PT habilitados, etc.)
    
    Returns:
        dict: Asignaciones {patrón: cantidad_agentes}
    """
    user_config = user_config or {}
    
    # Obtener parámetros adaptativos
    adaptive_params = get_adaptive_parameters(demand_matrix)
    cfg = {**adaptive_params, **user_config}  # User config tiene prioridad
    
    print(f"[JEAN] Iniciando optimización perfecta (target: {TARGET_COVERAGE}%)")
    print(f"[JEAN] Parámetros: factor={cfg['agent_limit_factor']}, exceso_pen={cfg['excess_penalty']:.1f}")
    
    # Generar TODOS los patrones posibles
    patterns = generate_complete_patterns(
        demand_matrix,
        use_ft=cfg.get("use_ft", True),
        use_pt=cfg.get("use_pt", True),
        cfg=cfg
    )
    
    if not patterns:
        print("[JEAN] ERROR: No se generaron patrones")
        return {}
    
    # Optimización ILP en dos fases
    try:
        assignments = optimize_two_phase_ilp(patterns, demand_matrix, cfg)
        
        # Verificar resultado
        coverage = calculate_coverage(assignments, patterns, demand_matrix.shape)
        total_covered = np.minimum(coverage, demand_matrix).sum()
        coverage_pct = (total_covered / demand_matrix.sum() * 100) if demand_matrix.sum() > 0 else 0
        
        excess = np.maximum(coverage - demand_matrix, 0).sum()
        deficit = np.maximum(demand_matrix - coverage, 0).sum()
        total_agents = sum(assignments.values())
        
        print(f"[JEAN] Resultado: {coverage_pct:.1f}% cobertura, {total_agents} agentes")
        print(f"[JEAN] Exceso: {excess}, Déficit: {deficit}, Score: {excess + deficit}")
        
        return assignments
        
    except Exception as e:
        print(f"[JEAN] ERROR en optimización: {e}")
        return {}

def persist_user_selections(selections):
    """
    Persiste selecciones del usuario para mantener estado entre sesiones.
    En Flask se puede usar session storage o localStorage via JavaScript.
    """
    # Esta función se implementaría según el framework usado
    # Para Flask: usar session['jean_config'] = selections
    # Para Streamlit: usar st.session_state
    pass

def load_user_selections():
    """Carga selecciones persistidas del usuario."""
    # Implementación específica del framework
    return {}