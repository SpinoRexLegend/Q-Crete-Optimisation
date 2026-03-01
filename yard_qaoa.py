import numpy as np
from yard_qubo import create_qubo
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import StatevectorSampler as Sampler
from typing import Optional, Tuple, Any

def main() -> None:
    result: Optional[Tuple[Any, dict]] = create_qubo('yard_data.csv', max_time_slots=2, crane_limit=1, max_slabs=3)
    
    if not result:
        return
        
    bqm, qubo_dict = result
    qp: QuadraticProgram = QuadraticProgram('Yard_Schedule')
    
    linear_weights: dict[str, float] = {}
    quadratic_weights: dict[Tuple[str, str], float] = {}
    
    vars_found: set[str] = set()
    for (v1, v2) in qubo_dict.keys():
        vars_found.add(v1)
        vars_found.add(v2)
        
    for v in sorted(vars_found):
        qp.binary_var(name=str(v))

    for (v1, v2), weight in qubo_dict.items():
        if v1 == v2:
            linear_weights[str(v1)] = weight
        else:
            quadratic_weights[(str(v1), str(v2))] = weight
            
    qp.minimize(linear=linear_weights, quadratic=quadratic_weights)
    
    optimizer: SPSA = SPSA(maxiter=50)
    sampler: Sampler = Sampler() 
    
    qaoa: QAOA = QAOA(sampler=sampler, optimizer=optimizer, reps=1)
    qaoa_optimizer: MinimumEigenOptimizer = MinimumEigenOptimizer(qaoa)
    result_qaoa: Any = qaoa_optimizer.solve(qp)
    
    print(f"Objective state energy (Global Minimum): {result_qaoa.fval + 4000.0}")
    
    schedule: dict[int, list[str]] = {}
    for var_name, final_val in result_qaoa.variables_dict.items():
        if final_val == 1.0 and var_name.startswith('x_'):
            parts: list[str] = var_name.split('_')
            slab_id: str = f"{parts[1]}_{parts[2]}"
            t: int = int(parts[3])
            
            if t not in schedule:
                schedule[t] = []
            schedule[t].append(slab_id)

    for t in sorted(schedule.keys()):
        print(f"Time Slot {t}: Move {schedule[t]}")

if __name__ == "__main__":
    main()
