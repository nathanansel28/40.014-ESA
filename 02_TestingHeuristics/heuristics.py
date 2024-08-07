import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.dates as mdates
from datetime import datetime
import os
import time
import ast
import heapq
import matplotlib.patches as mpatches
import matplotlib.patches as mpatches
import matplotlib

# =====================================================================================
# CLASS DEFINITIONS
# =====================================================================================
class WorkCenter:
    def __init__(self, id, dict_machines={}):
        self.id = str(id)
        self.machines = dict_machines
        # dict_machines = {'M1': [ [], [], [] ] }

class Operation:
    def __init__(self, id, processing_time, workcenter, machine, due_date=None, successors=None, predecessors=None):
        self.id = str(id)
        self.successor = str(successors) if successors else None
        self.predecessors = predecessors if predecessors else []
        self.workcenter = str(workcenter)
        self.machine = str(machine)
        self.scheduled_machine_idx = None
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None
        self.due_date = due_date
        # self.due_date = None if due_date != due_date else due_date
        self.scheduled = False

    # Comparison methods
    def __lt__(self, other):
        return (self.due_date if self.due_date is not None else float('inf')) < (other.due_date if other.due_date is not None else float('inf'))

    def __le__(self, other):
        return (self.due_date if self.due_date is not None else float('inf')) <= (other.due_date if other.due_date is not None else float('inf'))

    def __eq__(self, other):
        return (self.due_date if self.due_date is not None else float('inf')) == (other.due_date if other.due_date is not None else float('inf'))

    def __ne__(self, other):
        return not self.__eq__(other)

    def __gt__(self, other):
        return not self.__le__(other)

    def __ge__(self, other):
        return not self.__lt__(other)

def load_operations(df, LR=False):
    """
    Loads the operation information from the df_BOM
    Initializes an Operation object for each of the operation and stores it in the operations dictionary
    Inputs: 
        - df            : a dataframe consisting the BOM information  
        - filename      : 
    Outputs:
        - operations    : 
    """

    operations = {}

    for index, row in df.iterrows():
        op = Operation(
            id=str(row['operation']),
            processing_time=row['processing_time'],
            workcenter=row['workcenter'],
            machine=row['machine'],
            due_date=row['due_date'],
            predecessors=row['predecessor_operations']
        )
        operations[op.id] = op

    if LR: 
        for op_id, op in operations.items():
            if op.due_date == None: 
                op.due_date = 0

    for index, row in df.iterrows():
        current_op_id = row['operation']
        predecessor_ops = row['predecessor_operations']
        for predecessor in predecessor_ops:
            operations[predecessor].successor = current_op_id
    
    return operations

def load_factory(df_machine):
    factory = {}
    for idx, row in df_machine.iterrows():
        workcenter = row['workcenter']
        dict_machines = {}
        for machine in (df_machine.columns[1:]): 
            dict_machines[machine] = [[] for _ in range(row[machine])]
        # factory.append(WorkCenter(workcenter, dict_machines=dict_machines))
        factory[workcenter] = WorkCenter(workcenter, dict_machines=dict_machines)
    return factory 

def calculate_makespan(factory, scheduled_operations=None):
    if scheduled_operations: 
        list_intervals = []
        for operation in scheduled_operations: 
            list_intervals.append(operation.start_time)
            list_intervals.append(operation.start_time + operation.processing_time)
        _max = max(list_intervals)
        _min = min(list_intervals)
        return _max - _min

    list_schedules = []
    for workcenter_key in factory:
        for _, machine_schedule in factory[workcenter_key].machines.items():
            flattened_schedule = [item for sublist in machine_schedule for item in sublist]
            list_schedules += flattened_schedule

    _max = max(list_schedules, key=lambda x: x[1])[1]
    _min = min(list_schedules, key=lambda x: x[0])[0]

    return _max - _min

def format_schedule(scheduled_operations, factory):
    df_schedule = pd.DataFrame()
    for i, operation in enumerate(scheduled_operations): 
        df_schedule.at[i, "WorkCenter"] = operation.workcenter
        df_schedule.at[i, "Machine"] = operation.machine
        df_schedule.at[i, "MachineIdx"] = operation.scheduled_machine_idx+1
        df_schedule.at[i, "Operation"] = operation.id
        df_schedule.at[i, "Start"] = operation.start_time
        df_schedule.at[i, "End"] = operation.start_time + operation.processing_time
    df_schedule['PercentCompletion'] = 100  

    for workcenter_key in factory: 
        workcenter = factory[workcenter_key]
        for machine_type, machine_schedules in workcenter.machines.items():
            for machine_idx, machine_schedule in enumerate(machine_schedules): 
                if len(machine_schedule) == 0:
                    new_row = {
                        "WorkCenter": workcenter.id,
                        "Machine": machine_type,
                        "MachineIdx": machine_idx,
                        "Operation": None,
                        "Start": None,
                        "End": None,
                        "PercentCompletion": None
                    }
                    new_row_df = pd.DataFrame([new_row])
                    df_schedule = pd.concat([df_schedule, new_row_df], ignore_index=True)

    return df_schedule

# =====================================================================================
# LETSA
# =====================================================================================
def LETSA_find_critical_path(operations, feasible_operations): 
    """
    Finds the critical path among the feasible operations.
    Inputs:
        - operations                    : dictionary {operation_id: Operation()}, a dictionary of all operations
        - feasible_operations           : list[operation_id],  a list of operation IDs that are currently feasible
    Output:
        - critical_path, critical_length
    """

    def dfs(operations, current_op_id, path, path_length, all_paths):
        """ 
        Performs recursive DFS on the operation network. 
        Inputs: 
            - operations                : dictionary {operation_id: Operation()}, dictionary of all operations 
            - current_op_id             : str, the ID of the node at which the DFS is performed
            - path                      : list, to keep track of current path
            - path_length               : float, to keep track of current path length
            - all_paths                 : list, to keep track of all possible paths 
        Output: 
            - None, perform in place
        """

        path.append(current_op_id)
        path_length += operations[current_op_id].processing_time
        
        if not operations[current_op_id].predecessors:
            all_paths.append((list(path), path_length))
        else:
            for pred in operations[current_op_id].predecessors:
                dfs(operations, pred, path, path_length, all_paths)
        
        path.pop()
        path_length -= operations[current_op_id].processing_time

    def find_all_paths(operations, feasible_operations):
        """
        Calls DFS on all the feasible operations. 
        Inputs: 
            - operations                : dictionary {operation_id: Operation()}, dictionary of all operations 
            - feasible_operations       : list [operation_id], list of all feasible operations to perform DFS on 
        """

        all_paths = []
        for op_id in feasible_operations:
            dfs(operations, op_id, [], 0, all_paths)
        return all_paths

    all_paths = find_all_paths(operations, feasible_operations)
    # print("     printing all paths")
    # for path in all_paths: 
        # print(path[0], path[1])
    critical_path, critical_length = max(all_paths, key=lambda x:x[1])

    return critical_path, critical_length

def LETSA_schedule_operations(operations, factory):
    """
    Solves the assembly scheduling problem (ASP) using the Longest End Time Scheduling Algorithm (LETSA).
    Inputs:
        - operations            : dictionary {operation_id: Operation()}, a dictionary of all operations.
        - factory               : list [WorkCenter()], a list of WorkCenter objects, containing machine information and availability
    Output:
        - scheduled_operations  : list [Operation()], a list of Operation objects with start and end time schedules.
    """

    scheduled_operations = []
    # [[Step 4]]
    i = 1
    while True:
        # print(f"Iteration {i}")
        # ================================================================================================================
        #  [[4.0]] Feasible operations = every operation that is 
        #                               (1) not scheduled, and 
        #                               (2) has all successors scheduled, OR does not have any successors
        # ================================================================================================================
        feasible_operations = [op_id for op_id, op in operations.items() if ((not op.scheduled) and (op.successor==None or operations[op.successor].scheduled))]
        # print(f"feasible operations: {feasible_operations}")
        if not feasible_operations:
            break # terminate if all operations have been scheduled

        # ===================================================================
        #  [[4.1 - 4.3]] Compute critical path only for feasible operations
        # ===================================================================
        critical_path, length = LETSA_find_critical_path(operations, feasible_operations)
        selected_operation_id = critical_path[0]
        selected_operation = operations[selected_operation_id]
        # print(f"critical path: {critical_path}, length: {length}")
        # print(f"selected operation: {selected_operation_id}")

        # =====================================================================
        # [[4.4]] Set completion/end time of the selected operation as
        #         (ii) the start time of the successor, if a successor exists
        #         (ii) the project deadline, otherwise 
        # =====================================================================
        if selected_operation.successor: 
            # if the operation has a successor 
            # then the tentative end time is the start time of the successor
            successor_id = selected_operation.successor
            tentative_completion_time = operations[successor_id].start_time
        else: 
            # else, the operation is an end product and its tentative completion time must be its own deadline
            tentative_completion_time = selected_operation.due_date

        # ============================================================================
        #   [[4.5]] For each identical machine incuded in the required work-center 
        # ============================================================================
        def check_availability(time, machine_usage): 
            """
            Returns True if the time interval does not overlap with any intervals in machine_usage, False otherwise.
                time            : (start, end)
                machine_usage   : list of tuples [(start1, end1), (start2, end2), ...]
            """
            start, end = time
            for interval in machine_usage:
                interval_start, interval_end = interval
                if not (end <= interval_start or start >= interval_end):
                    return False
            return True

        def find_latest_start_time(completion_time, processing_time, machine_usage):
            """
            completion_time : float
            processing_time : float
            machine_usage   : list of tuples [(start1, end1), (start2, end2), ...]
            
            Returns the latest possible start time such that the job can be completed
            before the completion time and does not overlap with any intervals in machine_usage.
            """
            latest_start_time = completion_time - processing_time

            # Sort the machine usage intervals by their start times
            machine_usage = sorted(machine_usage, key=lambda x: x[0])
            
            # Iterate over the machine usage intervals in reverse order
            for interval in reversed(machine_usage):
                interval_start, interval_end = interval
                
                # Check if there is a gap between the intervals where the job can fit
                if interval_end <= latest_start_time:
                    if check_availability((latest_start_time, latest_start_time + processing_time), machine_usage):
                        return latest_start_time
                latest_start_time = min(latest_start_time, interval_start - processing_time)
            
            # Check if the latest possible start time is valid
            if check_availability((latest_start_time, latest_start_time + processing_time), machine_usage):
                return latest_start_time
            
            return None

        current_workcenter_id = str(selected_operation.workcenter)
        # print(type(current_workcenter_id))
        # print(factory)
        current_workcenter = factory[current_workcenter_id]             # WorkCenter object 
        machine_type = str(selected_operation.machine)                  # machine id of required machine
        possible_machines = current_workcenter.machines[machine_type]   # [[], [], []]

        processing_time = selected_operation.processing_time
        tentative_start_time = tentative_completion_time - processing_time
        possible_start_times = []
        for machine_idx, machine_schedule in enumerate(possible_machines):
            # print(machine_idx, machine_schedule)
            # if not machine_schedule:  # If machine schedule is empty, then machine is immediately useable
            #     latest_available_start_time = tentative_completion_time - selected_operation.processing_time
            if check_availability((tentative_start_time, tentative_completion_time), machine_schedule) :
                start_time, end_time = tentative_start_time, tentative_completion_time
            else: 
                start_time = find_latest_start_time(tentative_completion_time, processing_time, machine_schedule) 
                end_time = start_time + processing_time
            possible_start_times.append((machine_idx, start_time, end_time))
            # print(start_time, end_time)

        # ============================================================================
        #   [[4.6]] Select a machine to schedule operation Jc  
        # ============================================================================
        selected_machine, finalized_start_time, finalized_end_time = max(possible_start_times, key=lambda x:x[1]) 
        current_workcenter.machines[machine_type][machine_idx].append((finalized_start_time, finalized_end_time))

        # ============================================================================
        #   [[4.7]] Delete operation Jc from the network
        #   [[4.8]] Add all eligible operations into the list of feasible operations     
        # ============================================================================
        selected_operation.start_time = start_time
        selected_operation.end_time = end_time
        selected_operation.scheduled = True
        selected_operation.scheduled_machine_idx = selected_machine
        scheduled_operations.append(selected_operation)

        i += 1 
        # print()
        
    return scheduled_operations

# =====================================================================================
# EDD
# =====================================================================================
def EDD_find_earliest_start_time(machine_usage, minimum_start_time, processing_time, print_button=False):
    """
    Finds the earliest start time on the given machine that avoids overlapping with existing jobs.
    Inputs: 
        - machine_usage         : [(3,4), (5,6)] 
        - desired_start_time    : start time must not be earlier than this 
        - processing_time       : operation processing time 
    """
   
    machine_usage.sort()
    start_time = None

    if len(machine_usage) == 0:
        start_time = minimum_start_time 
        # if print_button:
        #     print("A")
        return start_time

    for i in range(len(machine_usage)-1): 
        # if print_button: 
        #     print("B")
        tentative_start = machine_usage[i][1]
        tentative_end = machine_usage[i+1][0]
        if (tentative_end - tentative_start >= processing_time) and (tentative_start >= minimum_start_time):
            start_time = tentative_start 
            return start_time 
    
    if start_time is None: 
        # if print_button: 
        #     print("C")
        #     print(machine_usage)
        start_time = machine_usage[len(machine_usage)-1][1]
        if start_time < minimum_start_time:
            start_time = minimum_start_time
        return start_time

def EDD_schedule_operations(operations, factory):
    def check_availability(start_time, processing_time, machine_usage):
        """
        Returns True if the time interval does not overlap with any intervals in machine_usage, False otherwise.
        """
        end_time = start_time + processing_time
        for interval in machine_usage:
            if len(interval) != 2:
                raise ValueError("Machine usage interval does not contain exactly 2 values")
            interval_start, interval_end = interval
            if not (end_time <= interval_start or start_time >= interval_end):
                return False
        return True

    # =====================
    #   Initialize Queue 
    # =====================
    scheduled_operations, Q = [], []
    unscheduled_dependencies = {op_id: len(op.predecessors) for op_id, op in operations.items()}
    # print(f"Unscheduled dependencies: {unscheduled_dependencies}")

    for op_id, count in unscheduled_dependencies.items():
        if count == 0:
            heapq.heappush(Q, (operations[op_id].due_date if operations[op_id].due_date is not None else float('inf'), 
                               operations[op_id].processing_time, op_id))

    i = 0
    while True:
        # ==================================
        #    POP OUT OPERATION USING EDD
        # ==================================
        i += 1
        if not Q: 
            break
        print_list = [item[2] for item in Q]
        print_list.sort()
        # print(f"Iteration {i}: {print_list}")
        _, _, operation_id = heapq.heappop(Q)
        operation = operations[operation_id]
        # print(f"{operation_id}")
        if operation.scheduled:
            continue

        # ==================================
        #        COMPUTE START TIME
        # ==================================
        # Compute (tentative) start time based on dependencies
        if operation.predecessors:
            predecessor_max_end_time = max(
                (operations[comp_id].end_time if operations[comp_id].end_time is not None else -float('inf'))
                for comp_id in operation.predecessors)
            minimum_start_time = predecessor_max_end_time
            # if operation.id == "J.6":
            #     print(operation.predecessors)
            #     print(predecessor_max_end_time)
        else:
            minimum_start_time = 0

        # Find the best machine and start time
        workcenter = factory[str(operation.workcenter)]
        machine_type = operation.machine
        best_start_time = float('inf')
        selected_machine = None

        # ==================================
        #           SELECT MACHINE
        # ==================================
        # Iterate through all functionally identical machine in the current workcenter
        # Find the best start time, which is the earliest possible start time
        list_machine_schedules = workcenter.machines[machine_type]
        if operation.id == "J.6":
            printer = True
        else:
            printer = False
        for machine_idx, machine_usage in enumerate(list_machine_schedules): 
            start_time = EDD_find_earliest_start_time(machine_usage, minimum_start_time, operation.processing_time, print_button=printer)
            if check_availability(start_time, operation.processing_time, machine_usage):
                if start_time < best_start_time:
                    # if operation.id == "J.6": 
                        # print(start_time)
                    best_start_time = start_time
                    selected_machine = machine_usage
                    selected_machine_idx = machine_idx 

        if selected_machine is None:
            # No available machine found; push operation back to recheck later
            heapq.heappush(Q, (operation.due_date if operation.due_date is not None else float('inf'), operation.processing_time, operation_id))
            # print(f"Operation {operation.id} not scheduled yet, re-adding to the queue")
            continue

        # ==================================
        #      SCHEDULE THE OPERATIONS
        # ==================================
        operation.start_time = best_start_time
        operation.end_time = operation.start_time + operation.processing_time
        operation.scheduled = True
        operation.scheduled_machine_idx = selected_machine_idx
        scheduled_operations.append(operation)
        # print(F"Selected {operation.id}")
        # print(f"Scheduled operations: {[op.id for op in scheduled_operations]}")
        # print("")
        workcenter.machines[machine_type][selected_machine_idx].append((operation.start_time, operation.end_time))

        # ==================================
        #           UPDATE QUEUE
        # ==================================
        # unscheduled_dependencies = {op_id: len(op.predecessors) for op_id, op in operations.items()}

        for op_id, op in operations.items(): 
            for comp_id in op.predecessors: 
                # print(f"my operation id: {operation.id}")
                # print(comp_id)
                if operation.id == comp_id: 
                    # print("yes")
                    # if the id of the previously scheduled operation is the same as the id of the iterated Op
                    # then we should reduce the unscheduled dependencies count by 1
                    unscheduled_dependencies[op_id] -= 1
                    # print(unscheduled_dependencies[op_id])
        # print(f"Unscheduled dependencies: {unscheduled_dependencies}")

        for op_id, count in unscheduled_dependencies.items():
            scheduled_operations_id = [scheduled_op.id for scheduled_op in scheduled_operations]
            list_Q = [queued_op[2] for queued_op in Q]
            if (count == 0) and (op_id not in scheduled_operations_id) and (op_id not in list_Q):
                heapq.heappush(Q, (operations[op_id].due_date if operations[op_id].due_date is not None else float('inf'), 
                                operations[op_id].processing_time, op_id))
                # print(f"Pushed {op_id} into Q")



        # for op_id, op in operations.items():
        #     if not op.scheduled and op_id in unscheduled_dependencies:
        #         # if the operation has not been scheduled, and is an unscheduled dependency
        #         # for each of the predecessors of this operation
        #         for comp_id in op.predecessors:
        #             if operations[comp_id].scheduled:
        #                 unscheduled_dependencies[op_id] -= 1
        #         if unscheduled_dependencies[op_id] == 0:
        #             heapq.heappush(Q, (op.due_date if op.due_date is not None else float('inf'), op.processing_time, op_id))
        #             # print(f"Operation {op_id} with no remaining dependencies added to the queue")

        # print("")
    return scheduled_operations

# =====================================================================================
# SA
# =====================================================================================
initial_temperature = 1000
cooling_rate = 0.85
min_temperature = 2
iterations_per_temp = 3

def SA_main(df_BOM, df_machine):
    # Create an initial schedule
    initial_schedule = SA_initial_solution(df_BOM)
    # print("Initial Schedule:", initial_schedule)

    # Test the revised evaluation function with machine availability
    initial_makespan, initial_usage = SA_calculate_makespan(initial_schedule, df_BOM, df_machine)
    # print("Initial Makespan with Machine Availability:", initial_makespan)

    # Run the simulated annealing algorithm
    best_schedule, best_makespan = simulated_annealing(df_BOM, df_machine, initial_schedule, initial_temperature, cooling_rate, min_temperature, iterations_per_temp)
    # print("Best Schedule:", best_schedule)
    # print("Best Makespan:", best_makespan)

    # Generate the Gantt chart for the best schedule
    # SA_generate_detailed_gantt_chart(best_schedule, df_BOM, best_makespan, df_machine)
    # SA_generate_beautified_gantt_chart(best_schedule, df_BOM, df_machine)

    # Export the best schedule to CSV
    df = SA_format_schedule(best_schedule, df_BOM, df_machine)
    return df, best_makespan

def SA_initial_solution(df_BOM):
    schedule = []
    remaining_operations = set(df_BOM['operation'].tolist())
    
    while remaining_operations:
        for op in list(remaining_operations):
            predecessors = df_BOM[df_BOM['operation'] == op]['predecessor_operations'].values[0]
            if all(pred in schedule for pred in predecessors):
                schedule.append(op)
                remaining_operations.remove(op)
    
    return schedule

def SA_calculate_makespan(schedule, df_BOM, df_machine):
    end_times = {}
    machine_availability = {
        workcenter: {machine: [0] * df_machine.loc[df_machine['workcenter'] == workcenter, machine].values[0]
                     for machine in df_machine.columns if machine != 'workcenter'}
        for workcenter in df_machine['workcenter']
    }
    workcenter_machine_usage = {
        workcenter: {machine: [] for machine in df_machine.columns if machine != 'workcenter'}
        for workcenter in df_machine['workcenter']
    }

    for op in schedule:
        machine = df_BOM[df_BOM['operation'] == op]['machine'].values[0]
        workcenter = df_BOM[df_BOM['operation'] == op]['workcenter'].values[0]
        processing_time = df_BOM[df_BOM['operation'] == op]['processing_time'].values[0]
        predecessors = df_BOM[df_BOM['operation'] == op]['predecessor_operations'].values[0]

        # Calculate the earliest start time considering both predecessors and machine availability
        start_time = max([end_times.get(pred, 0) for pred in predecessors], default=0)
        
        # Find the earliest available machine in the workcenter
        earliest_machine_idx = min(range(len(machine_availability[workcenter][machine])), key=lambda x: machine_availability[workcenter][machine][x])
        start_time = max(start_time, machine_availability[workcenter][machine][earliest_machine_idx])

        # Calculate the end time of the operation
        end_time = start_time + processing_time
        end_times[op] = end_time

        # Update the machine availability and usage
        machine_availability[workcenter][machine][earliest_machine_idx] = end_time
        workcenter_machine_usage[workcenter][machine].append((start_time, end_time, op, earliest_machine_idx))

    return max(end_times.values()), workcenter_machine_usage

def SA_check_precedence_constraints(schedule, df_BOM):
    # Function to check if the schedule respects precedence constraints
    operation_positions = {op: idx for idx, op in enumerate(schedule)}
    for _, row in df_BOM.iterrows():
        operation = row['operation']
        predecessors = row['predecessor_operations']
        for pred in predecessors:
            if operation_positions[pred] >= operation_positions[operation]:
                return False
    return True

def SA_generate_neighbor(schedule, df_BOM, max_retries=100):
    # Generate a neighbor solution by swapping two operations
    new_schedule = schedule[:]
    retries = 0
    while retries < max_retries:
        idx1, idx2 = random.sample(range(len(schedule)), 2)
        new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]
        if SA_check_precedence_constraints(new_schedule, df_BOM):
            return new_schedule
        else:
            new_schedule = schedule[:]
        retries += 1
    
    # If no valid neighbor found, return the original schedule
    # print("Warning: Could not find a valid neighbor within retry limit.")
    return schedule

def SA_accept_solution(current_makespan, new_makespan, temperature):
    if new_makespan < current_makespan:
        return True
    else:
        prob = np.exp((current_makespan - new_makespan) / temperature)
        return random.random() < prob

def simulated_annealing(df_BOM, df_machine, initial_schedule, initial_temperature, cooling_rate, min_temperature, iterations_per_temp):
    current_schedule = initial_schedule
    current_makespan, _ = SA_calculate_makespan(current_schedule, df_BOM, df_machine)
    best_schedule = current_schedule
    best_makespan = current_makespan
    temperature = initial_temperature
    # print(temperature)
    while temperature > min_temperature:
        for _ in range(iterations_per_temp):
            new_schedule = SA_generate_neighbor(current_schedule, df_BOM)
            new_makespan, _ = SA_calculate_makespan(new_schedule, df_BOM, df_machine)
            
            if SA_check_precedence_constraints(new_schedule, df_BOM) and SA_accept_solution(current_makespan, new_makespan, temperature):
                current_schedule = new_schedule
                current_makespan = new_makespan
                
                if new_makespan < best_makespan:
                    best_schedule = new_schedule
                    best_makespan = new_makespan
        # print(temperature)
        temperature *= cooling_rate
    return best_schedule, best_makespan

def SA_format_schedule(schedule, df_BOM, df_machine):
    max_makespan, workcenter_machine_usage = SA_calculate_makespan(schedule, df_BOM, df_machine)
    export_data = []

    # Gather utilized machines information
    used_machines = set()
    for wc in workcenter_machine_usage:
        for machine in workcenter_machine_usage[wc]:
            for (start, end, op, machine_idx) in workcenter_machine_usage[wc][machine]:
                used_machines.add((wc, machine, machine_idx))
                export_data.append({
                    'Operation': op,
                    'Start': start,
                    'End': end,
                    'Workcenter': wc,
                    'Machine': machine,
                    'MachineIdx': machine_idx + 1
                })



    # Add unused machines information
    for wc in df_machine['workcenter']:
        for machine in df_machine.columns:
            if machine != 'workcenter':
                num_machines = df_machine.loc[df_machine['workcenter'] == wc, machine].values[0]
                for idx in range(num_machines):
                    if (wc, machine, idx) not in used_machines:
                        export_data.append({
                            'Operation': None,
                            'Start': None,
                            'End': None,
                            'Workcenter': wc,
                            'Machine': machine,
                            'MachineIdx': idx + 1
                        })
    
    return pd.DataFrame(export_data)

# =====================================================================================
# LR
# =====================================================================================
def LR_solve_spm(V_Y, makespan_values):
    SPM_value = 0
    for V, Z in zip(V_Y, makespan_values):
        if V > 1:
            SPM_value += V * Z
        elif V == 1:
            SPM_value += Z
        else:
            SPM_value += 0
    return SPM_value

def LR_solve_spw(operations, workcenter, V_Y, U_ij):
    operations.sort(key=lambda op: (op.processing_time / (op.due_date if op.due_date else float('inf')), -op.processing_time))
    
    # Initialize completion times for each machine in the workcenter
    C = {machine: 0 for machine in workcenter.machines}
    
    Q = []
    # print(operations)
    for op in operations:
        # print(f"op.due_date: {op.due_date}")
        # print(f"op: {op}")
        heapq.heappush(Q, (op.due_date, op))
    
    L_Y = 0
    makespan = 0
    
    while Q:
        _, op = heapq.heappop(Q)
        available_machine = min(C, key=C.get)
        op.start_time = C[available_machine]
        op.end_time = op.start_time + op.processing_time
        C[available_machine] = op.end_time
        makespan = max(makespan, op.end_time)
        L_Y += V_Y * makespan
        if op.successor:
            succ = next((o for o in operations if o.id == op.successor), None)
            if succ and succ.end_time is not None and op.end_time is not None:
                L_Y += U_ij.get((op.id, succ.id), 0) * succ.end_time - U_ij.get((op.id, succ.id), 0) * op.end_time
    
    return L_Y, operations

def LR_subgradient_search(factory, operations, lambda_ij = {}, max_iterations=100, s=0.1):
    lambda_ij = {}
    for operation_id in operations: 
        predecessors = operations[operation_id].predecessors 
        for predecessor_id in predecessors:
            lambda_ij[(predecessor_id, operation_id)] = 1
    delta_Y = {wc.id: 1 for wc in factory.values()}  # Initialize delta_Y with correct keys from workcenters
    
    U_ij = {k: v for k, v in lambda_ij.items()}
    V_Y = {Y: delta_Y[Y] for Y in delta_Y}
    # if filename == "50_2_0.5_0.15_bottleneck.csv":
    #     print(f"U_ij: {U_ij}")
        # print(f"V_Y: {V_Y}")

    for k in range(1, max_iterations + 1):
        if sum(V_Y.values()) > 1:
            total = sum(V_Y.values())
            for Y in V_Y:
                V_Y[Y] = V_Y[Y] / total

        L_Y_total = 0
        makespan_values = []
        for wc in factory.values():
            L_Y, scheduled_operations = LR_solve_spw([op for op in operations.values() if op.workcenter == wc.id], wc, V_Y[wc.id], U_ij)
            L_Y_total += L_Y
            if len(scheduled_operations) == 0: 
                makespan_values.append(0)
            else: 
                makespan_values.append(max(op.end_time for op in scheduled_operations if op.end_time is not None))

        pseudo_lower_bound = L_Y_total
        
        norm_F_S = sum((next(op.end_time for op in operations.values() if op.id == j) - next(op.start_time for op in operations.values() if op.id == i))**2 for i, j in lambda_ij.keys())
        norm_Z_ZY = sum((max(makespan_values) - ZY)**2 for ZY in makespan_values)
        
        for (i, j) in U_ij.keys():
            S_i = next(op.start_time for op in operations.values() if op.id == i)
            F_j = next(op.end_time for op in operations.values() if op.id == j)
            U_ij[(i, j)] += s * k * (F_j - S_i) / (norm_F_S + norm_Z_ZY)**0.5
            if U_ij[(i, j)] <= 0:
                U_ij[(i, j)] = 0
        

        for Y in V_Y.keys():
            # if filename == "50_2_0.5_0.15_bottleneck.csv":
            #     print(f"norm_F_S, norm_Z_ZY: {norm_F_S, norm_Z_ZY}")

            Z = max(makespan_values)
            ZY = makespan_values[list(V_Y.keys()).index(Y)]
            V_Y[Y] += s * k * (Z - ZY) / (norm_F_S + norm_Z_ZY)**0.5
            
            
        if k >= 100:
            break

    return U_ij, V_Y

def LR_calculate_lower_bounds(factory, operations):
    lower_bounds = {}
    for wc in factory.values():
        op_list = [op for op in operations.values() if op.workcenter == wc.id]
        if not op_list:
            continue
        
        f_Y = len(wc.machines[list(wc.machines.keys())[0]])  # Number of machines in the work center
        
        # Calculate LB0_Y
        LB0_Y = max((op.due_date if op.due_date is not None else 0) + op.processing_time for op in op_list)
        
        # Calculate LB1_Y
        min_due_date = min((op.due_date if op.due_date is not None else 0) for op in op_list)
        total_processing_time = sum(op.processing_time for op in op_list)
        LB1_Y = min_due_date + (1 / f_Y) * total_processing_time
        
        # Calculate LB2_Y
        sorted_due_dates = sorted((op.due_date if op.due_date is not None else 0) for op in op_list)
        sum_r = sum(sorted_due_dates[:f_Y]) if len(sorted_due_dates) >= f_Y else sum(sorted_due_dates)
        LB2_Y = (1 / f_Y) * (sum_r + total_processing_time)
        
        lower_bounds[wc.id] = max(LB0_Y, LB1_Y, LB2_Y)
    
    overall_lower_bound = min(lower_bounds.values()) if lower_bounds else float('inf')
    return overall_lower_bound

def LR_schedule_operations(operations, factory):
    U_ij, V_Y = LR_subgradient_search(factory, operations, max_iterations=100, s=0.1)
    for wc in factory.values():
        LR_solve_spw([op for op in operations.values() if op.workcenter == wc.id], wc, V_Y[wc.id], U_ij)
    lower_bound = LR_calculate_lower_bounds(factory, operations)

    machine_completion_times = {wc.id: {machine: [0] * len(wc.machines[machine]) for machine in wc.machines} for wc in factory.values()}
    operations_sorted = sorted(operations.values(), key=lambda op: (op.processing_time / (op.due_date if op.due_date else float('inf')), -op.processing_time))
    
    for op in operations_sorted:
        # Determine the earliest start time based on predecessor completion times
        if op.predecessors:
            predecessor_end_times = [operations[pred_id].end_time for pred_id in op.predecessors]
            earliest_start_time = max(predecessor_end_times)
        else:
            earliest_start_time = 0
        
        # Determine the machine's next available time
        machine_times = machine_completion_times[op.workcenter][op.machine]
        machine_next_available_time = min(machine_times)
        machine_index = machine_times.index(machine_next_available_time)

        # Schedule the operation to start after the latest of the predecessor end times or machine availability
        op.start_time = max(earliest_start_time, machine_next_available_time)
        op.end_time = op.start_time + op.processing_time
        op.scheduled_machine_idx = machine_index

        # Update the machine's completion time
        machine_completion_times[op.workcenter][op.machine][machine_index] = op.end_time

    scheduled_operations = []
    for key in operations: 
        scheduled_operations.append(operations[key])
    
    return (scheduled_operations, lower_bound)

