# source-code
This repository contains MATLAB and Python code developed for the study titled: A sustainable multi-objective economic production quantity model with learning–forgetting and green technology investment

# Readme.txt
Description of the files and how to use the code


## Contents


# MATLAB
- CCT_Objectives_Non_rework.m (Objective function for non-rework production cycle under CCT policy)
- CCT_Objectives_Rework.m (Objective function for rework production cycle under CCT policy)
- Non_Rework_Cycle_NSGA2.m (NSGA-II algorithm for non-rework production cycle)
- Rework_Cycle_NSGA2.m (NSGA-II algorithm for rework production cycle)
- Non_Rework_Cycle_SPEA2.m (SPEA2 algorithm for non-rework production cycle)
- Rework_Cycle_SPEA2.m (SPEA2 algorithm for rework production cycle)
- skill_level_tpki.m (Computation of skill level for production rate in next cycle)
- skill_level_tdki.m (Computation of skill level for defective rate in next cycle)
- compute_A_bar.m (Computation of accumulated defective inventory between production cycles)
- topsis.m (Selects the best compromise solution from the Pareto front using TOPSIS method)


# Python
- normalized_hypervolume.py (Computes the normalized hypervolume using the Pymoo package)
- pareto_front.csv (Pareto front data exported from MATLAB containing objective values f1 and f2)


## Requirements
- MATLAB R2022b or later  
- Optimization Toolbox  
- Python 3.12 or later  
- Pymoo package (install using `pip install pymoo`)


## How to Run
1. Compute skill levels for the production and defective rates by running `skill_level_tpki.m` and `skill_level_tdki.m`.
2. Open the MATLAB script corresponding to the selected production cycle (Rework or Non-Rework) and algorithm (NSGA-II or SPEA2).
3. Run the script to load the parameters and execute the multi-objective optimization.
4. For **NSGA-II**:
   - The output objective values will be stored in `fp1` (first objective) and `fp2` (second objective).
5. For **SPEA2**:
   - The output objective values will be stored in `f1` (first objective) and `f2` (second objective).
6. Plot the Pareto front for the chosen configuration and save the results into a file named `pareto_front.csv`.
7. Run the Python script `normalized_hypervolume.py` to compute the hypervolume (HV), front length (L), and normalized hypervolume (NHV) metrics from the saved Pareto front.
8. To select the best compromise solution from the Pareto front, open and run `topsis.m` in MATLAB, providing it with the objective values as input.


### Optional:
You may modify algorithm parameters such as population size, generations, or learning coefficients within the MATLAB scripts to analyze different experimental scenarios.
