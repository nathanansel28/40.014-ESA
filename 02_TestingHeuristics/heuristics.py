import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# LETSA
import matplotlib.dates as mdates
from datetime import datetime
import os
import time
import ast
from functools import cache

# SA
import matplotlib.patches as mpatches

# EDD
import heapq

# ==============
#     LETSA
# ==============
class LETSAWorkCenter:
    def __init__(self, id, dict_machines={}):
        self.id = str(id)
        self.machines = dict_machines
        # dict_machines = {'M1': [ [], [], [] ] }

class LETSAOperation:
    def __init__(self, id, processing_time, workcenter, machine, due_date=None, successors=None, predecessors=None):
        self.id = str(id)
        self.successor = str(successors) if successors else None
        self.predecessors = predecessors if predecessors else []
        self.workcenter = str(workcenter)
        self.machine = str(machine)
        self.processing_time = processing_time
        self.start_time = None
        self.end_time = None
        self.due_date = None if due_date != due_date else due_date
        self.scheduled = False

def LETSA_load_operations(df, filename=None):
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
        op = LETSAOperation(
            id=str(row['operation']),
            processing_time=row['processing_time'],
            workcenter=row['workcenter'],
            machine=row['machine'],
            due_date=row['due_date'],
            predecessors=row['predecessor_operations']
        )
        operations[op.id] = op

    for index, row in df.iterrows():
        current_op_id = row['operation']
        predecessor_ops = row['predecessor_operations']
        for predecessor in predecessor_ops:
            operations[predecessor].successor = current_op_id
    
    return operations

def LETSA_load_factory(df_machine):
    factory = {}
    for idx, row in df_machine.iterrows():
        workcenter = row['workcenter']
        dict_machines = {}
        for machine in (df_machine.columns[1:]): 
            dict_machines[machine] = [[] for _ in range(row[machine])]
        # factory.append(WorkCenter(workcenter, dict_machines=dict_machines))
        factory[workcenter] = LETSAWorkCenter(workcenter, dict_machines=dict_machines)
    return factory 

def LETSA_plot_gantt_chart(scheduled_operations, plot_path=None, plot_name=None):   
    fig, ax = plt.subplots(figsize=(20, 20))

    # Get unique work centers
    workcenters = list(set(str(op.workcenter) for op in scheduled_operations))
    # print(f"Unique workcenters: {workcenters}")
    # print(workcenters)

    # Generate colors for each work center
    available_colors = plt.cm.tab20.colors
    num_colors = len(available_colors)
    colors = {workcenter: available_colors[i % num_colors] for i, workcenter in enumerate(workcenters)}
    # print(f"Colors dictionary: {colors}")

    for op in scheduled_operations:
        workcenter = str(op.workcenter)
        # print(f"Processing operation {op.id} for workcenter {workcenter}")
        
        start = op.start_time
        end = op.end_time
        
        if workcenter not in colors:
            print(f"Workcenter {workcenter} not found in colors dictionary")
            continue
        
        ax.barh(workcenter, end - start, left=start, color=colors[workcenter], edgecolor='black')
        ax.text(start + (end - start) / 2, workcenter, op.id, ha='center', va='center', color='black')

    ax.set_xlabel('Time')
    ax.set_ylabel('Work Center')
    ax.set_title('Gantt Chart of LETSA schedule')
    ax.set_yticks(range(len(workcenters)))
    ax.set_yticklabels(workcenters)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    if plot_path: 
        plt.savefig(f"{plot_path}//{plot_name}.png")
        plt.close(fig)
        plt.clf()


# ==============
#     EDD
# ==============
class EDDOperation:
    def __init__(self, id, predecessor_operations, end_product, due_date, processing_time, workcenter, machine):
        self.id = id
        self.predecessor_operations = predecessor_operations
        self.end_product = end_product
        self.due_date = due_date
        self.processing_time = processing_time
        self.workcenter = workcenter
        self.machine = machine
        self.start_time = None
        self.end_time = None
        self.scheduled = False

class EDDWorkCenter:
    def __init__(self, id, dict_machines={}):
        self.id = id
        self.machines = dict_machines

def EDD_load_operations(df):
    operations = {}
    for _, row in df.iterrows():
        op_id = row['operation']
        operations[op_id] = EDDOperation(
            id=op_id,
            processing_time=row['processing_time'],
            workcenter=row['workcenter'],
            machine=row['machine'],
            predecessor_operations=row['predecessor_operations'] if isinstance(row['predecessor_operations'], list) else [],
            end_product=row['end_product'],
            due_date=row['due_date']
        )
    return operations

def EDD_load_factory(df_machine):
    factory = {}
    for _, row in df_machine.iterrows():
        workcenter_id = row['workcenter']
        machines = {machine: [] for machine in df_machine.columns[1:] if row[machine] > 0}
        factory[workcenter_id] = EDDWorkCenter(id=workcenter_id, dict_machines=machines)
    return factory

def EDD_find_earliest_start_time(machine_usage, desired_start_time, processing_time):
    """
    Finds the earliest start time on the given machine that avoids overlapping with existing jobs.
    """
    # Ensure machine_usage is a list of tuples
    if not all(isinstance(interval, tuple) and len(interval) == 2 for interval in machine_usage):
        raise ValueError("Machine usage must be a list of tuples with exactly two values each.")
    
    # Flatten the list of usage intervals
    flat_usage = [interval for interval in machine_usage]
    flat_usage.sort()
    
    # print(f"Machine usage: {flat_usage}")
    # print(f"Desired start time: {desired_start_time}, Processing time: {processing_time}")

    if not flat_usage:
        return desired_start_time

    for i in range(len(flat_usage) + 1):
        if i == 0:
            interval_start = 0
            interval_end = flat_usage[i][0] if len(flat_usage) > 0 else float('inf')
        elif i == len(flat_usage):
            interval_start = flat_usage[i - 1][1]
            interval_end = float('inf')
        else:
            interval_start = flat_usage[i - 1][1]
            interval_end = flat_usage[i][0]

        # print(f"Checking interval: ({interval_start}, {interval_end})")

        if desired_start_time >= interval_start and desired_start_time + processing_time <= interval_end:
            return desired_start_time

        if interval_start + processing_time <= interval_end:
            return interval_start

    return desired_start_time

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

    scheduled_operations = []
    Q = []
    unscheduled_dependencies = {op_id: len(op.predecessor_operations) for op_id, op in operations.items()}

    # Initialize the queue with operations that have no dependencies, sorted by due dates
    for op_id, count in unscheduled_dependencies.items():
        if count == 0:
            heapq.heappush(Q, (operations[op_id].due_date if operations[op_id].due_date is not None else float('inf'), operations[op_id].processing_time, op_id))

    while Q:
        _, _, operation_id = heapq.heappop(Q)
        operation = operations[operation_id]

        if operation.scheduled:
            continue

        # Compute start time based on dependencies
        if operation.predecessor_operations:
            max_end_time = max(
                (operations[comp_id].end_time if operations[comp_id].end_time is not None else -float('inf'))
                for comp_id in operation.predecessor_operations
            )
            operation.start_time = max_end_time
        else:
            operation.start_time = 0

        # Find the best machine and start time
        workcenter = factory[operation.workcenter]
        best_start_time = float('inf')
        selected_machine = None
        selected_machine_name = None
        
        for machine_name, machine_usages in workcenter.machines.items():
            # print(f"Machine: {machine_name}, Usages: {machine_usages}")
            
            # Ensure machine_usages is a list of tuples
            if not all(isinstance(interval, tuple) and len(interval) == 2 for interval in machine_usages):
                raise ValueError(f"Machine usages for {machine_name} in workcenter {operation.workcenter} must be a list of tuples with exactly two values each.")
                
            start_time = EDD_find_earliest_start_time(machine_usages, operation.start_time, operation.processing_time)
            if check_availability(start_time, operation.processing_time, machine_usages):
                if start_time < best_start_time:
                    best_start_time = start_time
                    selected_machine = machine_usages
                    selected_machine_name = machine_name

        if selected_machine is None:
            # No available machine found; push operation back to recheck later
            heapq.heappush(Q, (operation.due_date if operation.due_date is not None else float('inf'), operation.processing_time, operation_id))
            # print(f"Operation {operation.id} not scheduled yet, re-adding to the queue")
            continue

        # Schedule the operation
        operation.start_time = best_start_time
        operation.end_time = operation.start_time + operation.processing_time
        selected_machine.append((operation.start_time, operation.end_time))
        selected_machine.sort()

        # # Debug information
        # print(f"Operation {operation.id}: Scheduled from {operation.start_time} to {operation.end_time} on machine {selected_machine_name} in workcenter {operation.workcenter}")
        # print(f"Machine {selected_machine_name} in workcenter {operation.workcenter} usage after scheduling operation {operation.id}: {selected_machine}")

        operation.scheduled = True
        scheduled_operations.append(operation)

        # Update dependencies and add ready operations to the queue
        for op_id, op in operations.items():
            if not op.scheduled and op_id in unscheduled_dependencies:
                for comp_id in op.predecessor_operations:
                    if operations[comp_id].scheduled:
                        unscheduled_dependencies[op_id] -= 1
                if unscheduled_dependencies[op_id] == 0:
                    heapq.heappush(Q, (op.due_date if op.due_date is not None else float('inf'), op.processing_time, op_id))
                    # print(f"Operation {op_id} with no remaining dependencies added to the queue")

    # Check if all operations have been scheduled
    unscheduled_ops = [op_id for op_id, op in operations.items() if not op.scheduled]
    if unscheduled_ops:
        # print(f"Unscheduled operations remaining: {unscheduled_ops}")
        # Attempt to schedule remaining unscheduled operations
        for op_id in unscheduled_ops:
            operation = operations[op_id]
            # Compute start time based on dependencies
            if operation.predecessor_operations:
                max_end_time = max(
                    (operations[comp_id].end_time if operations[comp_id].end_time is not None else -float('inf'))
                    for comp_id in operation.predecessor_operations
                )
                operation.start_time = max_end_time
            else:
                operation.start_time = 0

            # Find the best machine and start time
            workcenter = factory[operation.workcenter]
            best_start_time = float('inf')
            selected_machine = None
            selected_machine_name = None
            
            for machine_name, machine_usages in workcenter.machines.items():
                # print(f"Machine: {machine_name}, Usages: {machine_usages}")
                
                # Ensure machine_usages is a list of tuples
                if not all(isinstance(interval, tuple) and len(interval) == 2 for interval in machine_usages):
                    raise ValueError(f"Machine usages for {machine_name} in workcenter {operation.workcenter} must be a list of tuples with exactly two values each.")
                    
                start_time = EDD_find_earliest_start_time(machine_usages, operation.start_time, operation.processing_time)
                if check_availability(start_time, operation.processing_time, machine_usages):
                    if start_time < best_start_time:
                        best_start_time = start_time
                        selected_machine = machine_usages
                        selected_machine_name = machine_name

            if selected_machine is None:
                # print(f"Operation {operation.id} could not be scheduled")
                continue

            # Schedule the operation
            operation.start_time = best_start_time
            operation.end_time = operation.start_time + operation.processing_time
            selected_machine.append((operation.start_time, operation.end_time))
            selected_machine.sort()

            # # Debug information
            # print(f"Operation {operation.id}: Scheduled from {operation.start_time} to {operation.end_time} on machine {selected_machine_name} in workcenter {operation.workcenter}")
            # print(f"Machine {selected_machine_name} in workcenter {operation.workcenter} usage after scheduling operation {operation.id}: {selected_machine}")

            operation.scheduled = True
            scheduled_operations.append(operation)

    return scheduled_operations


# ==============
#       SA
# ==============
initial_temperature = 1000
cooling_rate = 0.8
min_temperature = 1
iterations_per_temp = 10

def SA_main():
    # Create an initial schedule
    initial_schedule = SA_initial_solution(df_BOM)
    print("Initial Schedule:", initial_schedule)
    # Test the revised evaluation function with machine availability
    initial_makespan, initial_usage = SA_calculate_makespan(initial_schedule, df_BOM, df_machine)
    print("Initial Makespan with Machine Availability:", initial_makespan)

    # Run the simulated annealing algorithm
    best_schedule, best_makespan = simulated_annealing(df_BOM, initial_schedule, initial_temperature, cooling_rate, min_temperature, iterations_per_temp)
    print("Best Schedule:", best_schedule)
    print("Best Makespan:", best_makespan)

    # Generate the Gantt chart for the best schedule
    SA_generate_detailed_gantt_chart(best_schedule, df_BOM, best_makespan, df_machine)
    SA_generate_beautified_gantt_chart(best_schedule, df_BOM, df_machine)

    # Export the best schedule to CSV
    # export_schedule_to_csv(best_schedule, df_BOM, df_machine, export_name)

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
    operation_positions = {op: idx for idx, op in enumerate(schedule)}
    for op in schedule:
        predecessors = df_BOM[df_BOM['operation'] == op]['predecessor_operations'].values[0]
        if any(operation_positions[pred] >= operation_positions[op] for pred in predecessors):
            return False
    return True

def SA_generate_neighbor(schedule, df_BOM):
    new_schedule = schedule[:]
    while True:
        idx1, idx2 = random.sample(range(len(schedule)), 2)
        new_schedule[idx1], new_schedule[idx2] = new_schedule[idx2], new_schedule[idx1]
        if SA_check_precedence_constraints(new_schedule, df_BOM):
            break
        else:
            new_schedule = schedule[:]
    return new_schedule

def SA_accept_solution(current_makespan, new_makespan, temperature):
    if new_makespan < current_makespan:
        return True
    else:
        prob = np.exp((current_makespan - new_makespan) / temperature)
        return random.random() < prob

def simulated_annealing(df_BOM, initial_schedule, initial_temperature, cooling_rate, min_temperature, iterations_per_temp):
    current_schedule = initial_schedule
    current_makespan, _ = SA_calculate_makespan(current_schedule, df_BOM, df_machine)
    best_schedule = current_schedule
    best_makespan = current_makespan
    temperature = initial_temperature
    print(temperature)
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
        print(temperature)
        temperature *= cooling_rate
    return best_schedule, best_makespan

def SA_generate_detailed_gantt_chart(schedule, df_BOM, total_makespan, df_machine):
    _, workcenter_machine_usage = SA_calculate_makespan(schedule, df_BOM, df_machine)

    # Generate colors for each machine-workcenter combination
    unique_machines = df_BOM['machine'].unique()
    unique_workcenters = df_BOM['workcenter'].unique()
    color_palette = plt.cm.get_cmap('tab20', len(unique_machines) * len(unique_workcenters))
    color_index = 0
    colors = {}
    for wc in unique_workcenters:
        for machine in unique_machines:
            colors[(wc, machine)] = color_palette(color_index)
            color_index += 1

    # Plot the Gantt chart
    fig, ax = plt.subplots(figsize=(15, 10))

    y_ticks = []
    y_labels = []
    y_position = 0

    for wc in df_machine['workcenter']:
        for machine in df_machine.columns:
            if machine != 'workcenter':
                num_machines = df_machine.loc[df_machine['workcenter'] == wc, machine].values[0]
                for machine_idx in range(num_machines):
                    if (wc in workcenter_machine_usage) and (machine in workcenter_machine_usage[wc]):
                        for (start, end, op, used_machine_idx) in workcenter_machine_usage[wc][machine]:
                            if used_machine_idx == machine_idx:
                                ax.broken_barh([(start, end - start)], (y_position - 0.4, 0.8), facecolors=(colors[(wc, machine)]))
                                ax.text(start, y_position, f" {op} ({machine}-{machine_idx + 1})", va='center', ha='left')

                    y_ticks.append(y_position)
                    y_labels.append(f"{wc} {machine}-{machine_idx + 1}")
                    y_position += 1

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine - Workcenter')
    ax.grid(True)

    # Ensure the x-axis matches the total makespan
    ax.set_xlim(0, total_makespan)

    # Create a legend for the color-coded machine and workcenter
    legend_patches = [mpatches.Patch(color=colors[(wc, machine)], label=f'{wc} - {machine}')
                      for wc in unique_workcenters for machine in unique_machines]
    ax.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.show()

def SA_generate_beautified_gantt_chart(schedule, df_BOM, df_machine):
    schedule_data = []
    operation_end_times = {}
    machine_availability = {
        workcenter: {machine: [0] * df_machine.loc[df_machine['workcenter'] == workcenter, machine].values[0]
                     for machine in df_machine.columns if machine != 'workcenter'}
        for workcenter in df_machine['workcenter']
    }

    # Define a light color palette for each workcenter
    unique_workcenters = df_BOM['workcenter'].unique()
    color_palette = plt.cm.get_cmap('Pastel1', len(unique_workcenters))
    workcenter_colors = {workcenter: color_palette(i) for i, workcenter in enumerate(unique_workcenters)}

    for op in schedule:
        machine = df_BOM[df_BOM['operation'] == op]['machine'].values[0]
        workcenter = df_BOM[df_BOM['operation'] == op]['workcenter'].values[0]
        processing_time = df_BOM[df_BOM['operation'] == op]['processing_time'].values[0]
        predecessors = df_BOM[df_BOM['operation'] == op]['predecessor_operations'].values[0]

        # Calculate the earliest start time considering predecessors
        start_time = max([operation_end_times.get(pred, 0) for pred in predecessors], default=0)
        
        # Find the earliest available machine in the workcenter
        earliest_machine_idx = min(range(len(machine_availability[workcenter][machine])), key=lambda x: machine_availability[workcenter][machine][x])
        start_time = max(start_time, machine_availability[workcenter][machine][earliest_machine_idx])

        # Calculate the end time of the operation
        end_time = start_time + processing_time
        schedule_data.append({
            'Operation': op,
            'Start': start_time,
            'End': end_time,
            'Machine': machine,
            'Workcenter': workcenter,
            'Color': workcenter_colors[workcenter],
            'MachineIdx': earliest_machine_idx + 1
        })

        # Update the availability time for the machine in the workcenter
        machine_availability[workcenter][machine][earliest_machine_idx] = end_time
        operation_end_times[op] = end_time

    # Convert schedule data to DataFrame for plotting
    df_schedule = pd.DataFrame(schedule_data)

    # Plot the Gantt chart
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, row in df_schedule.iterrows():
        ax.broken_barh(
            [(row['Start'], row['End'] - row['Start'])],
            (idx - 0.4, 0.8),
            facecolors=(row['Color'])
        )
        ax.text(
            row['Start'] + 0.1,
            idx,
            f"{row['Operation']} ({row['Machine']}-{row['MachineIdx']})",
            va='center',
            ha='left',
            color='black'
        )

    ax.set_yticks(range(len(df_schedule)))
    ax.set_yticklabels(df_schedule['Operation'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Operations')
    ax.grid(True)

    # Ensure the x-axis matches the total makespan
    ax.set_xlim(0, df_schedule['End'].max())

    plt.show()
















