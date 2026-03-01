import time
import streamlit as st
import pandas as pd
from typing import Any, List, Dict, Tuple
from physics_engine import PhysicsEngine
from quantum_optimizer import QuantumOptimizer, build_formulation
from dashboard_ui import DashboardUI

st.set_page_config(page_title="Team StarScream Q-Crete", layout="wide", initial_sidebar_state="expanded")

st.title("# Q-Crete Optimiser | Team StarScream")

@st.cache_data
def load_data() -> pd.DataFrame:
    df: pd.DataFrame = pd.read_csv('yard_data.csv')
    return df

def main() -> None:
    df: pd.DataFrame = load_data()
    
    physics_engine: PhysicsEngine = PhysicsEngine()
    quantum_optimizer: QuantumOptimizer = QuantumOptimizer()
    dashboard_ui: DashboardUI = DashboardUI()
    
    sidebar_params: Dict[str, Any] = dashboard_ui.render_sidebar(df)
    
    if sidebar_params["total_slabs"] > 10 or sidebar_params["crane_limit"] > 10:
        st.error("Input Constraints Exceeded. Total Slabs must be <= 10 and No. of Cranes must be <= 10 for QAOA Simulation bounds.")
        st.stop()
    
    dynamic_slabs: pd.DataFrame = physics_engine.predict_strength(
        df=sidebar_params["base_slabs"], 
        temp_mod=sidebar_params["temp_mod"], 
        hum_mod=sidebar_params["hum_mod"]
    )
    
    eligible_slabs: List[str] = dashboard_ui.render_pinn_results(dynamic_slabs, sidebar_params["strength_threshold"])
    
    st.divider()
    st.header("2. Optimizer Dispatch")
    
    if len(eligible_slabs) > 0:
        _, qubo_latex = build_formulation(eligible_slabs, sidebar_params["crane_limit"])
        dashboard_ui.render_latex_early(qubo_latex)
    else:
        st.warning("No slabs are ready to move under current weather conditions!")
        return
        
    if st.button("Submit Optimization"):
        total_opt_start: float = time.time()
        with st.spinner("Initializing Solver Branch... (10s Timeout)"):
            q_slabs: List[str] = eligible_slabs
            crane_limit: int = sidebar_params["crane_limit"]
            
            result_data: Dict[str, Any] = quantum_optimizer.optimize(q_slabs, crane_limit)
            
            dashboard_ui.render_qaoa_results(
                q_slabs, 
                result_data['schedule'], 
                result_data['elapsed'], 
                result_data['fval'],
                result_data['branch']
            )
            
            confidence_prob: float = dashboard_ui.render_probability_graph(result_data['samples'], result_data['best_bitstring'])
            
            dashboard_ui.render_executive_summary(
                q_slabs, crane_limit, 
                sidebar_params["weather_scenario"], 
                result_data['best_bitstring'], 
                confidence_prob
            )
            
        total_opt_time: float = time.time() - total_opt_start
        st.markdown(f"***\n_Optimization Time: {total_opt_time:.4f}s_")

if __name__ == "__main__":
    main()
