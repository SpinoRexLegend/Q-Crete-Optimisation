import pandas as pd
import dimod
from collections import defaultdict
import numpy as np
from typing import Optional, Tuple

def create_qubo(data_path: str, max_time_slots: int = 10, crane_limit: int = 2, strength_threshold: float = 70.0, max_slabs: int = 15) -> Optional[Tuple[dimod.BinaryQuadraticModel, dict]]:
    try:
        df: pd.DataFrame = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Could not find {data_path}. Please run generate_yard_data.py first.")
        return None

    eligible_slabs: list[str] = df[df['Strength'] > strength_threshold]['Slab_ID'].tolist()
    num_slabs: int = min(len(eligible_slabs), max_slabs)
    eligible_slabs = eligible_slabs[:num_slabs]
    
    bqm: dimod.BinaryQuadraticModel = dimod.BinaryQuadraticModel('BINARY')
    variables: dict[Tuple[str, int], str] = {}
    
    for i in eligible_slabs:
        for t in range(max_time_slots):
            variables[(i, t)] = f'x_{i}_{t}'

    penalty_assignment: float = 1000.0
    penalty_crane: float = 500.0
    reward_early: float = 10.0

    for i in eligible_slabs:
        slab_vars: list[str] = [variables[(i, t)] for t in range(max_time_slots)]
        bqm.add_linear_equality_constraint(
            [(var, 1.0) for var in slab_vars],
            constant=-1.0,
            lagrange_multiplier=penalty_assignment
        )

    for t in range(max_time_slots):
        slot_vars: list[str] = [variables[(i, t)] for i in eligible_slabs]
        slack_var_names: list[str] = []
        for s in range(crane_limit):
            slack_name: str = f'slack_t{t}_s{s}'
            slack_var_names.append(slack_name)
            bqm.add_variable(slack_name)

        constraint_terms: list[Tuple[str, float]] = [(var, 1.0) for var in slot_vars] + [(var, 1.0) for var in slack_var_names]
        
        bqm.add_linear_equality_constraint(
            constraint_terms,
            constant=-crane_limit,
            lagrange_multiplier=penalty_crane
        )

    for i in eligible_slabs:
        for t in range(max_time_slots):
            var: str = variables[(i, t)]
            bqm.add_linear(var, reward_early * t)

    qubo_dict, offset = bqm.to_qubo()
    
    return bqm, qubo_dict

if __name__ == "__main__":
    result: Optional[Tuple[dimod.BinaryQuadraticModel, dict]] = create_qubo('yard_data.csv', max_time_slots=8, crane_limit=2)
    
    if result:
        bqm, qubo = result
        import neal
        sampler: neal.SimulatedAnnealingSampler = neal.SimulatedAnnealingSampler()
        sampleset: dimod.SampleSet = sampler.sample(bqm, num_reads=50)
        
        best_sample: dict = sampleset.first.sample
        energy: float = sampleset.first.energy
        
        schedule: defaultdict = defaultdict(list)
        for var_name, final_val in best_sample.items():
            if final_val == 1 and var_name.startswith('x_'):
                parts: list[str] = var_name.split('_')
                slab_id: str = f"{parts[1]}_{parts[2]}"
                t: int = int(parts[3])
                schedule[t].append(slab_id)

        for t in sorted(schedule.keys()):
            print(f"Time Slot {t}: Move {schedule[t]} (Total cranes used: {len(schedule[t])})")
        
        print(f"\nFinal Energy State: {energy}")
