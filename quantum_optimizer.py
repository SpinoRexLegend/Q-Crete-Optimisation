import time
import numpy as np
import streamlit as st
from typing import Tuple, Any, Dict, List
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, MinimumEigenOptimizationResult
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler
from concurrent.futures import ThreadPoolExecutor, TimeoutError

def build_formulation(q_slabs: List[str], crane_limit: int) -> Tuple[QuadraticProgram, str]:
    num_slabs: int = len(q_slabs)
    target_moves: int = min(num_slabs, crane_limit)
    
    P: float = 1000.0
    
    Q: np.ndarray = np.zeros((num_slabs, num_slabs))
    c: np.ndarray = np.zeros(num_slabs)
    
    qp: QuadraticProgram = QuadraticProgram('Yard_Schedule')
    var_names: List[str] = []
    
    for i in range(num_slabs):
        var_name: str = f'x_{i}'
        var_names.append(var_name)
        qp.binary_var(name=var_name)
        
    for i in range(num_slabs):
        c[i] += P * (1.0 - 2.0 * target_moves)
        c[i] -= (10.0 + 0.1 * i)
        
        Q[i, i] += P
        
        for j in range(i + 1, num_slabs):
            Q[i, j] += 2.0 * P
            
    linear: Dict[str, float] = {var_names[i]: float(c[i]) for i in range(num_slabs)}
    quadratic: Dict[Tuple[str, str], float] = {(var_names[i], var_names[j]): float(Q[i, j]) for i in range(num_slabs) for j in range(i + 1, num_slabs) if Q[i, j] != 0}
    
    qp.minimize(linear=linear, quadratic=quadratic)
    
    qubo_latex: str = f"H(x) = \\sum_{{i}} (-10) x_i + {P} \\left(\\sum_{{i}} x_i - {target_moves}\\right)^2"
    
    return qp, qubo_latex

@st.cache_data(show_spinner=False)
def solve_qaoa_cached(q_slabs: List[str], crane_limit: int) -> Dict[str, Any]:
    qp, qubo_latex = build_formulation(q_slabs, crane_limit)
    
    optimizer: SPSA = SPSA(maxiter=15)
    sampler: StatevectorSampler = StatevectorSampler()
    
    qaoa: QAOA = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    qaoa_optimizer: MinimumEigenOptimizer = MinimumEigenOptimizer(qaoa)
    
    start_time: float = time.time()
    result: MinimumEigenOptimizationResult = qaoa_optimizer.solve(qp)
    elapsed: float = time.time() - start_time
    
    num_slabs: int = len(q_slabs)
    best_bitstring: str = "".join([str(int(result.variables_dict.get(f'x_{i}', 0))) for i in range(num_slabs)])
    
    schedule: Dict[int, List[str]] = {0: []}
    for i in range(num_slabs):
        if result.variables_dict.get(f'x_{i}', 0) == 1.0:
            schedule[0].append(q_slabs[i])

    return {
        'schedule': schedule,
        'best_bitstring': best_bitstring,
        'elapsed': elapsed,
        'fval': result.fval,
        'samples': result.samples if hasattr(result, "samples") else None,
        'branch': 'Quantum QAOA',
        'qubo_latex': qubo_latex
    }

class QuantumOptimizer:
    def optimize(self, q_slabs: List[str], crane_limit: int) -> Dict[str, Any]:
        heuristic_fallback: Dict[str, Any] = self._build_heuristic(q_slabs, crane_limit)
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(solve_qaoa_cached, q_slabs, crane_limit)
            try:
                return future.result(timeout=10.0)
            except TimeoutError:
                return heuristic_fallback

    def _build_heuristic(self, q_slabs: List[str], crane_limit: int) -> Dict[str, Any]:
        start_time: float = time.time()
        schedule: Dict[int, List[str]] = {0: []}
        
        target_moves: int = min(len(q_slabs), crane_limit)
        best_bitstring: str = ""
        
        for i in range(len(q_slabs)):
            if i < target_moves:
                schedule[0].append(q_slabs[i])
                best_bitstring += "1"
            else:
                best_bitstring += "0"
                
        elapsed: float = time.time() - start_time
        return {
            'schedule': schedule,
            'best_bitstring': best_bitstring,
            'elapsed': elapsed,
            'fval': -1000.0 * target_moves,
            'samples': None,
            'branch': 'Classical Heuristic (Timeout Fallback)',
            'qubo_latex': "H(x) = Classical Heuristic Matrix Generated (Timeout > 10s limits exceeded)"
        }
