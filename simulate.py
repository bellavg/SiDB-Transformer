import os
import pandas as pd
from datetime import timedelta
from fiction.pyfiction import *

# path to the layouts folder
BENCH_ROOT = "/home/marcel/git/mnt-sidb-bench/random_layouts/"
# lower bound of SiDB concentration to simulate
LOWER_BOUND_SIDBS = 1
# upper bound of SiDB concentration to simulate (inclusive)
UPPER_BOUND_SIDBS = 10
# SiDB concentration after which ExGS doesn't need to bother
EXGS_DROPOUT = 15


def dot_accuracy(sim_result_exgs, sim_result_quicksim):
    # if QuickSim didn't find a solution, accuracy for that run is 0
    if len(sim_result_quicksim.charge_distributions) == 0:
        return 0

    exgs_cds = sim_result_exgs.charge_distributions[0]
    quicksim_cds = sim_result_quicksim.charge_distributions[0]

    num_sidbs = exgs_cds.num_cells()
    difference_counter = 0

    # for each SiDB
    for c in exgs_cds.cells():
        # if ExGS and QuickSim differ in their predictions
        if exgs_cds.get_charge_state(c) != quicksim_cds.get_charge_state(c):
            difference_counter += 1


    return (num_sidbs - difference_counter) / num_sidbs


def print_average_runtimes(runtimes):
    for n, time in runtimes.items():
        if time:
            average = sum(timedelta.total_seconds() for timedelta in time) / len(time)

            print(f"{n}: {average}")
            runtimes[n] = average


def print_average_accuracies(accuracies):
    for n, accuracy in accuracies.items():
        if accuracy:
            average = sum(accuracy) / len(accuracy)

            print(f"{n}: {average}")
            accuracies[n] = average


quicksim_runtimes = {}
quicksim_accuracies = {}

exgs_runtimes = {}
exgs_accuracies = {}

for n in range(1, UPPER_BOUND_SIDBS + 1):
    folder_name = f"number_sidbs_{n}"
    folder_path = os.path.join(BENCH_ROOT, folder_name, "sqd")

    if not os.path.exists(folder_path):
        continue

    print(f"Simulating SQD layouts with {n} SiDBs")

    qs_t = []
    exgs_t = []

    qs_acc = []
    exgs_acc = []

    for sqd_file in os.listdir(folder_path):
        # print(sqd_file)

        lyt = read_sqd_layout(os.path.join(folder_path, sqd_file))
        cds = charge_distribution_surface(lyt)

        sim_result_quicksim = quicksim(cds)
        qs_t.append(sim_result_quicksim.simulation_runtime)

        if n <= EXGS_DROPOUT:
            sim_result_exgs = exhaustive_ground_state_simulation(cds)
            exgs_t.append(sim_result_exgs.simulation_runtime)

            qs_acc.append(dot_accuracy(sim_result_exgs, sim_result_quicksim))
            exgs_acc.append(1)

    quicksim_runtimes[n] = qs_t
    quicksim_accuracies[n] = qs_acc

    exgs_runtimes[n] = exgs_t
    exgs_accuracies[n] = exgs_acc


print()
print("ExGS")
print("----")
print("n: average t in s")
print_average_runtimes(exgs_runtimes)
print("n: average accuracy")
print_average_accuracies(exgs_accuracies)

print()
print("QuickSim")
print("--------")
print("n: average t in s")
print_average_runtimes(quicksim_runtimes)
print("n: average accuracy")
print_average_accuracies(quicksim_accuracies)


# convert to pandas and save as csv
series1 = pd.Series(exgs_runtimes, name="ExGS runtimes")
series2 = pd.Series(exgs_runtimes, name="ExGS accuracies")
series3 = pd.Series(exgs_runtimes, name="QuickSim runtimes")
series4 = pd.Series(exgs_runtimes, name="QuickSim accuracies")

df = pd.concat([series1, series2, series3, series4], axis=1)
df.to_csv("simulation_data.csv")
