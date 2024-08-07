# Operations Scheduling

Assembly operations are integral to most manufacturing systems, involving the combination of multiple inputs to produce a final output. Examples include inserting electronic components into circuit boards, assembling body parts and engines into automobiles, and combining chemicals to produce other substances. These operations are characterised by their need for all necessary components to be present before processing can begin, which can complicate production flows. 

The Bill-of-Materials (BOM) plays a crucial role in assembly operations, detailing all the components required at each stage of the assembly process. A shortage of any component can disrupt the entire assembly line, affecting not only the assembly operations but all other related fabrication lines. To mitigate these disruptions, it is common to subordinate the scheduling and control of the fabrication lines to the assembly operations. This is achieved by specifying a final assembly schedule and working backward to schedule the fabrication lines. 

The primary goal of this project is to create a schedule that minimises the makespan (total time to complete all tasks) while considering constraints such as workstation availability, task order, and potential downtimes. The complexity of assembly line scheduling arises from the vast number of possible schedules, which grows exponentially with the number of tasks and workstations. Traditional optimization methods, while aiming for the absolute best solution, become computationally impractical for large-scale problems with numerous tasks and workstations. Hence, heuristic methods are preferred in this scenario.

# Directories
- **01_Data**: In this section, we gather and generate datasets crucial to our project.  
  -  01_BenchmarkDataset: This dataset is taken from [A benchmark dataset for multi-objective flexible job shop cell scheduling](https://data.mendeley.com/datasets/rtzby7pv7m/1).
  -  02_RemanufacturingDataset: This dataset is taken from [A New Remanufacturing System Scheduling Model](https://figshare.com/articles/dataset/A_New_Remanufacturing_System_Scheduling_Model_with_Diversified_Reprocessing_Routes_Using_a_Hybrid_Meta-heuristic_Algorithm/17026007?file=31488986).
  -  03_Randomized: This dataset was generated using our data generator (`data_generator_clean.ipynb`). It largely follows principles and methods shared in [(Hall & Posner, 2001)](https://doi.org/10.1287/opre.49.6.854.10014). 

- **02_TestingHeuristics**: Our efforts implementing and developing heuristics are documented here.
- **03_Heuristics**: Our working heuristics can be accessed via `heuristics.ipynb`. We developed four heuristics, namely: 
  - **Lead Time Evaluation and Scheduling Algorithm (LETSA)**: Developed by [(Agrawal et al., 1994)](https://doi.org/10.1080/15458830.1996.11770710), this heuristic produces a near-optimal operations schedule while minimizing makespan as well as WIP costs. It is a very efficient heuristic, requiring very little computational efforts.
  - **Earliest Due Date (EDD)**: Also known as the Moore-Hodgson algorithm, the EDD heuristic minimizes the number of tardy jobs by scheduling operations with the closest deadline first. While this algorithm is very common especially for single machine settings, our implementation allows the EDD to work in a multi-machine setting where tasks are constrained to certain precedence constraints.
  - **Simulated Annealing**: Simulated Annealing (SA) is an metaheuristic optimisation technique that iteratively improves a solution by exploring its neighborhood, avoiding local optima and converge on a high quality solution. This would allow us to obtain a near-optimal solution while thoroughly exploring the solution space.
  - **Lagrangian Relaxation**: Developed by [(Xu & Nagi, 2013)](https://doi.org/10.1109/tase.2013.2259816) , the Lagrangian Relaxation (LR) approach involves formulating the scheduling problem as a MILP and relaxing the constraints into the objective function with the use of appropriate Lagrangian multipliers and solving the decomposed subproblems to obtain feasible solutions. This algorithm provides a lower bound and minimizes the runtime and also obtain near-optimal results with a very short computational time.
 
# Working Web App

Our heuristic was implemented in a web app which can be found here: https://github.com/nathanansel28/40.014-ESA-webapp

# Contributions
Nathan Ansel \
Lee Peck Yeok \
Georgia Karen Lau \
Kong Le'ann Norah \
Oon Eu Kuan Eugene \
Tan Chong Hao \
Long Yan Ting

### Special Thanks
This project was done as part of the 2D project in the 40.014 Engineering Systems Architecture and 40.012 Manufacturing and Service Operations. We express our special thanks to Prof. Sun Zeyu and Prof. Rakesh Nagi for their guidance and support throughout the duration of the project. 

