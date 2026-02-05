import math
import random
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from copy import deepcopy
import gurobipy as gp
from gurobipy import GRB
import heapq


def rss_demands(demand, cover):
    return sum((cover[i] - demand[i]) ** 2 for i in range(len(demand)))


def twist_rounding_schema(all_keys, vals, dmnds, funcs):
    demands = deepcopy(dmnds)
    all_sets = []
    costs = []
    cnt_covered_demands = [0 for _ in range(len(dmnds))]

    for i in range(len(all_keys)):
        key = all_keys[i]

        for j in range(vals[i] + 1, len(funcs[i])):
            cost = funcs[i][j] - funcs[i][j - 1]
            tea = deepcopy(key)
            tmp = tea + (cost,)
            all_sets.append(tmp)
            costs.append(cost)

        if vals[i] == 0:
            continue

        for j in range(len(demands)):
            if key[j] == 0:
                continue
            demands[j] -= vals[i]
            cnt_covered_demands[j] += vals[i]

    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]
            heapq.heappush(heap, (ratio, i, cover[i]))

    used = [0 for _ in range(m)]

    total_weight = 0
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue
        if cover[i] == cover_when_inserted:
            total_weight += all_sets[i][n]
            used[i] = 1
            for j in range(n):
                if all_sets[i][j] == 1:
                    cnt_covered_demands[j] += 1
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))
    return total_weight, cnt_covered_demands


def lp_rounding_schema(all_keys, vals, dmnds, weights):
    demands = deepcopy(dmnds)
    all_sets = []
    costs = []
    cnt_covered_demands = [0 for _ in range(len(dmnds))]

    for i in range(len(all_keys)):
        key = all_keys[i]

        if vals[i] == 0:
            cost = weights[i]
            tea = deepcopy(key)
            tmp = tea + (cost,)
            all_sets.append(tmp)
            costs.append(cost)
            continue

        for j in range(len(demands)):
            if key[j] == 0:
                continue
            demands[j] -= vals[i]
            cnt_covered_demands[j] += vals[i]

    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]  # Weight / cover
            heapq.heappush(heap, (ratio, i, cover[i]))

    used = [0 for _ in range(m)]

    total_weight = 0
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue
        if cover[i] == cover_when_inserted:
            total_weight += all_sets[i][n]
            used[i] = 1
            for j in range(n):
                if all_sets[i][j] == 1:
                    cnt_covered_demands[j] += 1
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))

    return total_weight, cnt_covered_demands


def run_greedy(dic, dmnds):
    demands = deepcopy(dmnds)
    all_keys = dic.keys()
    all_sets = []
    cnt_covered_demands = [0 for _ in range(len(dmnds))]
    for key in all_keys:
        costs = deepcopy(dic[key])
        for cost in costs:
            tea = deepcopy(key)
            tmp = tea + (cost,)
            all_sets.append(tmp)

    residual = list(demands)
    n = len(demands)
    m = len(all_sets)
    cover = [sum(1 for j in range(n) if all_sets[i][j] == 1 and residual[j] > 0) for i in range(m)]
    covering_sets = [[] for _ in range(n)]
    for i in range(m):
        for j in range(n):
            if all_sets[i][j] == 1:
                covering_sets[j].append(i)
    heap = []
    for i in range(m):
        if cover[i] > 0:
            ratio = all_sets[i][n] / cover[i]
            heapq.heappush(heap, (ratio, i, cover[i]))

    used = [0 for _ in range(m)]

    total_weight = 0
    while heap:
        ratio, i, cover_when_inserted = heapq.heappop(heap)
        if used[i]:
            continue
        if cover[i] == cover_when_inserted:
            total_weight += all_sets[i][n]
            used[i] = 1
            for j in range(n):
                if all_sets[i][j] == 1:
                    cnt_covered_demands[j] += 1
                if all_sets[i][j] == 1 and residual[j] > 0:
                    residual[j] -= 1
                    if residual[j] == 0:
                        for k in covering_sets[j]:
                            if cover[k] > 0:
                                cover[k] -= 1
                                if cover[k] > 0:
                                    new_ratio = all_sets[k][n] / cover[k]
                                    heapq.heappush(heap, (new_ratio, k, cover[k]))
    return total_weight, cnt_covered_demands


def calc_pwl(functions, demands, all_keys):
    max_demand = []
    for key in all_keys:
        max_tmp = 0
        for i in range(len(demands)):
            if key[i] == 0:
                continue
            max_tmp = max(max_tmp, demands[i])
        max_demand.append(max_tmp + 1)

    knots_x_list = []
    knots_y_list = []
    for i in range(len(functions)):
        func = functions[i]
        curr_dems = max_demand[i]
        curr_size = min(len(func), curr_dems + 1)
        curr_func = deepcopy(func[:curr_size])

        pt = 0

        curr_x = [0]
        curr_y = [0]
        eps = 4.0
        while pt < curr_size - 1:
            base_value = curr_func[pt]
            pt2 = pt + 1
            while pt2 < curr_size and curr_func[pt2] <= base_value * eps:
                pt2 += 1
            pt2 -= 1
            opt = max(pt + 1, pt2)
            pt = opt
            curr_x.append(pt)
            curr_y.append(curr_func[pt])

        knots_x_list.append(curr_x)
        knots_y_list.append(curr_y)

    return knots_x_list, knots_y_list


def run_twist(dic2, demands):
    all_keys = list(dic2.keys())
    convex_functions, cnt = [], []
    for key in all_keys:
        cur_q = dic2[key]
        ps, curr_f = 0, [0]
        for x in cur_q:
            ps += x
            curr_f.append(ps)
        convex_functions.append(curr_f)
        cnt.append(len(curr_f) - 1)

    knots_x_list, knots_y_list = calc_pwl(convex_functions, demands, all_keys)
    n, m = len(demands), len(cnt)

    model = gp.Model("Set_Multicover")
    x = [model.addVar(lb=0, ub=cnt[i], vtype=GRB.CONTINUOUS, name=f"x_{i}") for i in range(m)]
    z = [model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"z_{i}") for i in range(m)]
    model.setObjective(gp.quicksum(z), GRB.MINIMIZE)

    for i in range(m):
        knots_x, knots_y = knots_x_list[i], knots_y_list[i]
        for k in range(len(knots_x) - 1):
            x1, x2 = knots_x[k], knots_x[k + 1]
            y1, y2 = knots_y[k], knots_y[k + 1]
            slope, intercept = (y2 - y1) / (x2 - x1), y1 - ((y2 - y1) / (x2 - x1)) * x1
            model.addConstr(z[i] >= slope * x[i] + intercept, f"cost_{i}_{k}")

    for j in range(n):
        model.addConstr(gp.quicksum(x[i] * all_keys[i][j] for i in range(m)) >= demands[j], f"demand_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    if model.Status != GRB.OPTIMAL:
        print(f"Problem status: {model.Status}")
        return None, None

    solution = [var.X for var in x]
    optimal_cost = model.ObjVal
    print("Status: OPTIMAL")
    print(optimal_cost)

    optans, new_vals = 0, []
    for i, sol in enumerate(solution):
        val = int(round(sol))
        new_vals.append(val)
        optans += convex_functions[i][val]

    rounding_cost, cnt_covered_demands = twist_rounding_schema(all_keys, new_vals, demands, convex_functions)
    print("Rounding cost:", rounding_cost)
    return optans + rounding_cost, cnt_covered_demands


def run_lp(data, demands):
    n = len(data)
    L = [tuple(data.iloc[i]) for i in range(n)]
    A = np.array([row[:-1] for row in L])
    weights = np.array([row[-1] for row in L])

    model = gp.Model("Set_Cover")
    x = [model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}") for i in range(n)]
    model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(n)), GRB.MINIMIZE)

    for j in range(A.shape[1]):
        model.addConstr(gp.quicksum(A[i][j] * x[i] for i in range(n)) >= demands[j], f"demand_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    opt_value = model.ObjVal
    weighted_sum, new_vals = 0, []
    for i in range(n):
        val = int(round(x[i].X))
        weighted_sum += weights[i] * val
        new_vals.append(val)

    print("LP solution:")
    print(weighted_sum)
    print(opt_value)

    rounding_cost, cnt_covered_demands = lp_rounding_schema(L, new_vals, demands, weights)
    print("Rounding cost:", rounding_cost)
    return weighted_sum + rounding_cost, cnt_covered_demands


def randomized_rounding_scheme(all_keys, lp_vals, demands, weights, alpha=2):
    n = len(all_keys)
    m = len(demands)
    demands_left = deepcopy(demands)
    cnt_covered_demands = [0 for _ in range(m)]
    total_cost = 0
    scale = alpha * math.log(max(2, m))
    probs = [min(1, lp_vals[i] * scale) for i in range(n)]
    rounded_vals = [1 if random.random() < probs[i] else 0 for i in range(n)]
    for i in range(n):
        if rounded_vals[i] == 1:
            total_cost += weights[i]
            key = all_keys[i]
            for j in range(m):
                if key[j] == 1:
                    cnt_covered_demands[j] += 1
                    demands_left[j] = max(0, demands_left[j] - 1)

    return total_cost, cnt_covered_demands


def run_lp_randomized_rounding(data, demands, alpha=2):
    n = len(data)
    L = [tuple(data.iloc[i]) for i in range(n)]
    A = np.array([row[:-1] for row in L])
    weights = np.array([row[-1] for row in L])

    model = gp.Model("Set_Cover")
    x = [model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}") for i in range(n)]
    model.setObjective(gp.quicksum(weights[i] * x[i] for i in range(n)), GRB.MINIMIZE)

    for j in range(A.shape[1]):
        model.addConstr(gp.quicksum(A[i][j] * x[i] for i in range(n)) >= demands[j], f"demand_{j}")

    model.setParam("OutputFlag", 0)
    model.optimize()

    opt_value = model.ObjVal
    lp_vals = [x[i].X for i in range(n)]

    print("LP Opt value:", opt_value)

    rounding_cost, cnt_covered_demands = randomized_rounding_scheme(L, lp_vals, demands, weights, alpha=alpha)
    print("SetCover-style randomized rounding cost:", rounding_cost)

    return rounding_cost, cnt_covered_demands


def construct_priority_queues(data):
    dic = defaultdict(list)
    h = 0
    for _, row in data.iterrows():
        h += 1
        key = tuple(row[:-1])
        priority = row["weight"]
        dic[key].append(priority)
    for key in dic.keys():
        dic[key].sort()
    return dic


def run_dp(data_dict, demands):
    """
    Returns: (min_weight, covered_demands)
    """
    num_dims = len(demands)
    target_demands = tuple(demands)

    # State Definition:
    # Key: tuple of CAPPED demands (used to merge equivalent sub-problems)
    # Value: (min_cost, tuple of UNCAPPED coverage)
    
    # Base case: Cost 0, Coverage 0
    initial_coverage = tuple([0.0] * num_dims)
    dp = {
        tuple([0] * num_dims): (0.0, initial_coverage)
    }

    # 1. Iterate over every group configuration (key in the dict)
    for group_contribs, weights_list in data_dict.items():
        
        # 2. Iterate over every available item of this configuration
        # (We treat multiple weights in the list as distinct items available to pick)
        for weight in weights_list:
            
            # Create a copy for the next layer (representing the "Don't Take" option)
            new_dp = dp.copy()

            # 3. Extend every existing path (The "Take" option)
            for state, (current_cost, current_coverage) in dp.items():
                
                new_state_capped = []
                new_coverage_uncapped = []
                
                for i in range(num_dims):
                    # Contribution of the current item to group i
                    contrib = group_contribs[i]
                    
                    # Track ACTUAL total coverage (Uncapped)
                    # This sums up exactly what you picked, matching the Greedy logic
                    new_coverage_uncapped.append(current_coverage[i] + contrib)
                    
                    # Track DP State (Capped)
                    # We cap this to ensure the dictionary size stays finite (DP optimization)
                    val = state[i] + contrib
                    if val > target_demands[i]:
                        val = target_demands[i]
                    new_state_capped.append(val)
                
                new_state_key = tuple(new_state_capped)
                new_total_coverage = tuple(new_coverage_uncapped)
                new_cost = current_cost + weight

                # 4. Update Logic
                # If we haven't reached this capped state before, OR we found a cheaper way to reach it:
                existing_entry = new_dp.get(new_state_key)
                
                if existing_entry is None or new_cost < existing_entry[0]:
                    new_dp[new_state_key] = (new_cost, new_total_coverage)
            
            # Move to next item
            dp = new_dp

    # 5. Retrieve result for the target demands
    # Note: Python dictionaries handle mixed int/float keys gracefully (1 == 1.0)
    result = dp.get(target_demands)

    if result is None:
        # If demands could not be met
        return math.inf, [0] * num_dims

    min_weight, final_coverage = result
    return min_weight, list(final_coverage)


import matplotlib.ticker as ticker


def vizualize(x, y1, y2, y3, y4, y5, y6, y7, y8, x_lable, y_lable_1, y_lable_2, dataset, exp_type):
    # === STRATEGY SELECTION ===
    # If "demands_custom_random", use Faceted Subplots (One plot per x-tick)
    if exp_type == "demands_custom_random":
        n_groups = len(x)
        # Create a row of subplots, one for each x-tick
        fig, axes = plt.subplots(1, n_groups, figsize=(16, 6), sharey=False, gridspec_kw={"wspace": 0.7})
        plt.rcParams.update({"font.size": 14})  # Slightly smaller font to fit side-by-side

        width = 0.2
        # Local x-coordinates for bars within a subplot (just 0, 1, 2, 3 centered)
        local_x = np.arange(1)

        # Iterate through each group and plot on its specific axis
        for i, ax in enumerate(axes):
            # Extract values for this specific group (i)
            vals = [y2[i], y3[i], y4[i], y1[i]]  # Greedy, 2+e, RR-LP, 2-approx

            # Plot bars on the subplot
            # Note: We plot at relative positions [-1.5, -0.5, 0.5, 1.5] around 0
            ax.bar(
                -1.5 * width,
                vals[0],
                width,
                color="lemonchiffon",
                hatch="/",
                edgecolor="black",
                label="Greedy" if i == 0 else "",
            )
            ax.bar(
                -0.5 * width,
                vals[1],
                width,
                color="lightsalmon",
                hatch="**",
                edgecolor="black",
                label="(2+ϵ)-approx." if i == 0 else "",
            )
            ax.bar(
                0.5 * width,
                vals[2],
                width,
                color="thistle",
                hatch="xx",
                edgecolor="black",
                label="RR-LP" if i == 0 else "",
            )
            ax.bar(
                1.5 * width,
                vals[3],
                width,
                color="lightblue",
                hatch="oo",
                edgecolor="black",
                label="2-approx." if i == 0 else "",
            )

            # === SOLUTION: Shorten Labels using Offset ===
            # This creates a formatter that pulls the large constant out.
            # Example: [1000005, 1000010] becomes ticks [5, 10] with "+1e6" at the top.
            fmt = ticker.ScalarFormatter(useOffset=True, useMathText=True)

            # Force scientific notation for anything larger than 10^3 or smaller than 10^-3
            fmt.set_powerlimits((-3, 3))

            ax.yaxis.set_major_formatter(fmt)

            # Optional: Decrease the number of ticks to max 3 or 4 to save vertical space
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))

            # Reduce padding between tick labels and axis
            ax.tick_params(axis="y", pad=2)

            # Formatting the individual subplot (The "Zoom")
            # 1. Set Title to the Demand Range
            range_label = r"$[2^{{{}}}, 2^{{{}}}]$".format(i + 1, i + 2) if i > 0 else r"$[0, 2^{1}]$"
            ax.set_xlabel(range_label, fontsize=14)
            # Remove the default title
            ax.set_title("")
            # 2. Tight Linear Scale (The key to seeing differences)
            y_min, y_max = min(vals), max(vals)
            margin = (y_max - y_min) * 0.5 if y_max != y_min else y_max * 0.1
            ax.set_ylim(y_min - margin, y_max + margin)

            # 3. Clean up ticks
            ax.set_xticks([])  # Remove x-ticks (the title is the label)

            # Optional: Rotate y-labels or make them smaller to prevent overlap
            ax.tick_params(axis="y", labelsize=14, rotation=45)

        # Global Labels
        fig.text(0.5, 0.02, x_lable, ha="center", fontsize=20)
        fig.text(0.02, 0.5, y_lable_1, va="center", rotation="vertical", fontsize=20)

        # Single Legend for the whole figure
        # fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=16)
        plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15)
        plt.savefig(f"plot/comparison_faceted_{exp_type}_{dataset}.pdf", bbox_inches="tight")
    else:
        # === STANDARD LOG PLOT (Your original code for other exp_types) ===
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({"font.size": 20})
        width = 0.2
        plt.bar(np.arange(len(x)) - 1.5 * width, y2, width, label="Greedy", color="white", hatch="/", edgecolor="black")
        plt.bar(
            np.arange(len(x)) - 0.5 * width,
            y3,
            width,
            label="(2+ϵ)-approx.",
            color="white",
            edgecolor="black",
            hatch="**",
        )
        plt.bar(np.arange(len(x)) + 0.5 * width, y4, width, label="RR-LP", color="white", edgecolor="black", hatch="xx")
        plt.bar(
            np.arange(len(x)) + 1.5 * width, y1, width, label="2-approx.", color="white", hatch="oo", edgecolor="black"
        )

        # ... (rest of your original xticks logic) ...
        if exp_type == "n":
            plt.xticks(
                np.arange(len(x)),
                labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
            )
        elif exp_type == "distribution":
            plt.xticks(np.arange(len(x)), labels=[x[i][:4] + "." for i in range(len(x))])
        else:
            plt.xticks(np.arange(len(x)), x, rotation=45)

        plt.xlabel(x_lable)
        plt.ylabel(y_lable_1)
        plt.yscale("log")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=16, frameon=True)
        plt.savefig(f"plot/comparison_varying_{exp_type}_{dataset}.pdf", bbox_inches="tight")

    # --- PLOT 2: LINE CHART (Unchanged logic, just cleanup) ---
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 20})

    plt.plot(x, y5, label="2-approx.", marker="o", linestyle="-", linewidth=2.5, markersize=8, color="blue")
    plt.plot(x, y6, label="Greedy", marker="^", linestyle="--", linewidth=2.5, markersize=8, color="orange")
    plt.plot(x, y7, label="(2+ϵ)-approx.", marker="x", linestyle="-.", linewidth=2.5, markersize=10, color="red")
    plt.plot(x, y8, label="RR-LP", marker="s", linestyle=":", linewidth=2.5, markersize=8, color="purple")

    if exp_type == "n":
        plt.xticks(
            np.arange(len(x)),
            labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(x_lable + " - log")
        plt.ylabel(y_lable_2 + " - log")
    elif exp_type == "demands_custom_random":  # Changed to elif to match logic structure
        plt.xticks(
            np.arange(len(x)),
            labels=[r"$[0, 2^{1}]$"] + [r"$[2^{{{}}}, 2^{{{}}}]$".format(i, i + 1) for i in range(1, len(x))],
            rotation=45,
        )
        plt.xscale("log") # Usually x is linear index for categories, careful with log scale on categorical x
        plt.yscale("log")
        plt.yscale("log")

        # --- THE FIX ---
        ax = plt.gca()

        # 1. Force the Major Locator to show all values (0.1, 0.2, ... 0.9)
        # 'subs="all"' promotes the numbers between 10 and 100 to be Major ticks
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, subs='all')) 

        # 2. Force the labels to be numbers ("20", "30") instead of scientific ("2x10^1")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

        # Optional: Remove the offset text (the "x10^2" at the top of the axis)
        ax.yaxis.get_major_formatter().set_scientific(False)
        plt.xlabel(x_lable)
        plt.ylabel(y_lable_2 + " - log")
        plt.minorticks_on()
    elif exp_type == "distribution":
        plt.xticks(np.arange(len(x)), labels=[x[i][:4] + "." for i in range(len(x))])
        plt.yscale("log")
        plt.xlabel(x_lable)
        plt.ylabel(y_lable_2 + " - log")
    else:
        plt.xticks(np.arange(len(x)), x)
        plt.yscale("log")
        plt.xlabel(x_lable)
        plt.ylabel(y_lable_2 + " - log")

    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=16, frameon=True)
    plt.savefig(f"plot/time_varying_{exp_type}_{dataset}.pdf", bbox_inches="tight")


# def vizualize(x, y1, y2, y3, y4, y5, y6, y7, y8, x_lable, y_lable_1, y_lable_2, dataset, exp_type):
#     plt.figure(figsize=(10, 6))
#     plt.rcParams.update({"font.size": 20})
#     width = 0.2
#     plt.bar(
#         np.arange(len(x)) - 1.5 * width,
#         y2,
#         width,
#         label="Greedy",
#         color="white",
#         hatch="/",
#         edgecolor="black",
#         linewidth=1,
#     )
#     plt.bar(
#         np.arange(len(x)) - 0.5 * width,
#         y3,
#         width,
#         label="(2+ϵ)-approx.",
#         color="white",
#         edgecolor="black",
#         hatch="**",
#         linewidth=1,
#     )
#     plt.bar(
#         np.arange(len(x)) + 1.5 * width,
#         y1,
#         width,
#         label="2-approx.",
#         color="white",
#         hatch="oo",
#         edgecolor="black",
#         linewidth=1,
#     )
#     plt.bar(
#         np.arange(len(x)) + 0.5 * width,
#         y4,
#         width,
#         label="RR-LP",
#         color="white",
#         edgecolor="black",
#         hatch="xx",
#         linewidth=1,
#     )

#     if exp_type == "n":
#         plt.xticks(
#             np.arange(len(x)),
#             labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
#         )

#     if exp_type == "demands_custom_random":
#         plt.xticks(
#             np.arange(len(x)),
#             labels = [r"$[0, 2^{1}]$"] + [r"$[2^{{{}}}, 2^{{{}}}]$".format(i, i + 1) for i in range(1, len(x))],
#             rotation=45
#         )
#         plt.yscale("log")
#     elif exp_type == "distribution":
#         plt.xticks(np.arange(len(x)), labels=[x[i][:4] + "." for i in range(len(x))])
#     else:
#         plt.xticks(np.arange(len(x)), x, rotation=45)
#     plt.xlabel(x_lable)
#     plt.ylabel(y_lable_1)
#     plt.yscale("log")
#     plt.grid(True, axis="y", linestyle="--", alpha=0.7)
#     # plt.legend()
#     plt.savefig(f"plot/comparison_varying_{exp_type}_{dataset}.pdf", bbox_inches="tight")

#     plt.figure(figsize=(10, 6))
#     plt.rcParams.update({"font.size": 20})

#     plt.plot(x, y5, label="2-approx.", marker="o", linestyle="-", linewidth=2.5, markersize=8, color="blue")
#     plt.plot(x, y6, label="Greedy", marker="^", linestyle="--", linewidth=2.5, markersize=8, color="orange")
#     plt.plot(x, y7, label="(2+ϵ)-approx.", marker="x", linestyle="-.", linewidth=2.5, markersize=10, color="red")
#     plt.plot(x, y8, label="RR-LP", marker="s", linestyle=":", linewidth=2.5, markersize=8, color="purple")

#     if exp_type == "n":
#         plt.xticks(
#             np.arange(len(x)),
#             labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
#         )
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.xlabel(x_lable + " - log")
#         plt.ylabel(y_lable_2 + " - log")
#     if exp_type == "demands_custom_random":
#         plt.xticks(
#             np.arange(len(x)),
#             labels = [r"$[0, 2^{1}]$"] + [r"$[2^{{{}}}, 2^{{{}}}]$".format(i, i + 1) for i in range(1, len(x))],
#             rotation=45
#         )
#         plt.xscale("log")
#         plt.yscale("log")
#         plt.xlabel(x_lable)
#         plt.ylabel(y_lable_2 + " - log")
#     elif exp_type == "distribution":
#         plt.xticks(np.arange(len(x)), labels=[x[i][:4] + "." for i in range(len(x))])
#         plt.yscale("log")
#         plt.xscale("linear")
#         plt.xlabel(x_lable)
#         plt.ylabel(y_lable_2 + " - log")
#     else:
#         plt.xticks(np.arange(len(x)), x)
#         plt.yscale("log")
#         plt.xscale("linear")
#         plt.xlabel(x_lable)
#         plt.ylabel(y_lable_2 + " - log")
#     plt.grid(True, linestyle="--", alpha=0.7)
#     # plt.legend()
#     plt.savefig(f"plot/time_varying_{exp_type}_{dataset}.pdf", bbox_inches="tight")


def visualize_rss(x, y1, y2, y3, y4, dataset):
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 16})

    plt.plot(x, y1, label="2-approx.", marker="o", linestyle="-", linewidth=2.5, markersize=8, color="blue")
    plt.plot(x, y2, label="Greedy", marker="^", linestyle="--", linewidth=2.5, markersize=8, color="orange")
    plt.plot(x, y3, label="(2+ϵ)-approx.", marker="x", linestyle="-.", linewidth=2.5, markersize=10, color="red")
    plt.plot(x, y4, label="RR-LP", marker="s", linestyle=":", linewidth=2.5, markersize=8, color="purple")

    plt.xlabel("Number of Items")
    plt.ylabel("RSS")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=14, frameon=True)

    plt.savefig(f"plot/rss_varying_group_{dataset}.pdf", bbox_inches="tight")
    plt.close()
    
    
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import math

def vizualize_2(x, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, x_lable, y_lable_1, y_lable_2, dataset, exp_type):
    """
    y1: 2-approx Cost
    y2: Greedy Cost
    y3: (2+e)-approx Cost
    y4: RR-LP Cost
    y5: 2-approx Time
    y6: Greedy Time
    y7: (2+e)-approx Time
    y8: RR-LP Time
    y9: DP Cost (New) - List may be shorter than others
    y10: DP Time (New) - List may be shorter than others
    """
    
    # Common settings
    dp_color = "palegreen"
    dp_hatch = "++"
    dp_label = "DP (Exact)"

    # === STRATEGY SELECTION ===
    if exp_type == "demands_custom_random":
        n_groups = len(x)
        fig, axes = plt.subplots(1, n_groups, figsize=(18, 6), sharey=False, gridspec_kw={"wspace": 0.7})
        plt.rcParams.update({"font.size": 14})

        # Reduced width to fit bars
        width = 0.15 
        
        for i, ax in enumerate(axes):
            # 1. Define standard strategies first
            vals = [y2[i], y3[i], y4[i], y1[i]] 
            offsets = [-2, -1, 0, 1]
            colors = ["lemonchiffon", "lightsalmon", "thistle", "lightblue"]
            hatches = ["/", "**", "xx", "oo"]
            labels = ["Greedy", "(2+ϵ)-approx.", "RR-LP", "2-approx."]

            # 2. Conditionally add DP if data exists for this index
            if i < len(y9):
                vals.append(y9[i])
                offsets.append(2)
                colors.append(dp_color)
                hatches.append(dp_hatch)
                labels.append(dp_label)
            
            # 3. Plot the bars
            for j, (val, offset, col, hat, lab) in enumerate(zip(vals, offsets, colors, hatches, labels)):
                ax.bar(
                    offset * width,
                    val,
                    width,
                    color=col,
                    hatch=hat,
                    edgecolor="black",
                    label=lab if i == 0 else "",
                )

            # === FORMATTING ===
            fmt = ticker.ScalarFormatter(useOffset=True, useMathText=True)
            fmt.set_powerlimits((-3, 3))
            ax.yaxis.set_major_formatter(fmt)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3))
            ax.tick_params(axis="y", pad=2)

            # Title / Axis Labels
            range_label = r"$[2^{{{}}}, 2^{{{}}}]$".format(i + 1, i + 2) if i > 0 else r"$[0, 2^{1}]$"
            ax.set_xlabel(range_label, fontsize=14)
            ax.set_title("")

            # Tight Linear Scale
            valid_vals = [v for v in vals if v != math.inf]
            if valid_vals:
                y_min, y_max = min(valid_vals), max(valid_vals)
                margin = (y_max - y_min) * 0.5 if y_max != y_min else (y_max * 0.1 if y_max != 0 else 1.0)
                ax.set_ylim(max(0, y_min - margin), y_max + margin)

            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=14, rotation=45)

        # Global Labels
        fig.text(0.5, 0.02, x_lable, ha="center", fontsize=20)
        fig.text(0.02, 0.5, y_lable_1, va="center", rotation="vertical", fontsize=20)

        plt.subplots_adjust(left=0.08, right=0.98, top=0.85, bottom=0.15)
        plt.savefig(f"plot/comparison_faceted_{exp_type}_{dataset}.pdf", bbox_inches="tight")

    else:
        # === STANDARD LOG PLOT ===
        plt.figure(figsize=(10, 6))
        plt.rcParams.update({"font.size": 20})
        
        width = 0.15
        
        # Plotting bars
        # y9 is handled automatically here because len(y9) restricts the range of arange
        plt.bar(np.arange(len(x)) - 2.0 * width, y2, width, label="Greedy", color="lemonchiffon", hatch="/", edgecolor="black")
        plt.bar(np.arange(len(x)) - 1.0 * width, y3, width, label="(2+ϵ)-approx.", color="lightsalmon", edgecolor="black", hatch="**")
        plt.bar(np.arange(len(x)) + 0.0 * width, y4, width, label="RR-LP", color="thistle", edgecolor="black", hatch="xx")
        plt.bar(np.arange(len(x)) + 1.0 * width, y1, width, label="2-approx.", color="lightblue", hatch="oo", edgecolor="black")
        
        # DP: uses len(y9) so it stops creating bars when data runs out
        plt.bar(np.arange(len(y9)) + 2.0 * width, y9, width, label=dp_label, color="palegreen", hatch=dp_hatch, edgecolor="black")

        # X-Ticks Logic
        if exp_type == "n":
            plt.xticks(
                np.arange(len(x)),
                labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
            )
        elif exp_type == "distribution":
            plt.xticks(np.arange(len(x)), labels=[str(val)[:4] + "." for val in x])
        else:
            plt.xticks(np.arange(len(x)), x, rotation=45)

        plt.xlabel(x_lable)
        plt.ylabel(y_lable_1)
        plt.yscale("log")
        plt.grid(True, axis="y", linestyle="--", alpha=0.7)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=16, frameon=True)
        plt.savefig(f"plot/comparison_varying_{exp_type}_{dataset}.pdf", bbox_inches="tight")

    # --- PLOT 2: LINE CHART (TIME) ---
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 20})

    # Existing Lines
    plt.plot(x, y5, label="2-approx.", marker="o", linestyle="-", linewidth=2.5, markersize=8, color="blue")
    plt.plot(x, y6, label="Greedy", marker="^", linestyle="--", linewidth=2.5, markersize=8, color="orange")
    plt.plot(x, y7, label="(2+ϵ)-approx.", marker="x", linestyle="-.", linewidth=2.5, markersize=10, color="red")
    plt.plot(x, y8, label="RR-LP", marker="s", linestyle=":", linewidth=2.5, markersize=8, color="purple")
    
    # DP Line: Use x[:len(y10)] so it uses the correct X coordinates but stops early
    plt.plot(x[:len(y10)], y10, label=dp_label, marker="D", linestyle="-", linewidth=2.5, markersize=8, color="green")

    # X-Ticks Logic for Time Plot
    if exp_type == "n":
        plt.xticks(
            np.arange(len(x)),
            labels=[r"$2^{{{}}}$".format(i) for i in range(int(math.log2(x[0])), int(math.log2(x[-1])) + 1)],
        )
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(x_lable + " - log")
        plt.ylabel(y_lable_2 + " - log")
    elif exp_type == "demands_custom_random":
        plt.xticks(
            np.arange(len(x)),
            labels=[r"$[0, 2^{1}]$"] + [r"$[2^{{{}}}, 2^{{{}}}]$".format(i, i + 1) for i in range(1, len(x))],
            rotation=45,
        )
        plt.yscale("log")
        plt.xlabel(x_lable)
        plt.ylabel(y_lable_2 + " - log")
    elif exp_type == "distribution":
        plt.xticks(np.arange(len(x)), labels=[str(val)[:4] + "." for val in x])
        plt.yscale("log")
        plt.xlabel(x_lable)
        plt.ylabel(y_lable_2 + " - log")
    else:
        plt.xticks(x)
        plt.yscale("log")
        plt.xlabel(x_lable)
        plt.ylabel(y_lable_2 + " - log")

    plt.grid(True, linestyle="--", alpha=0.7)
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=16, frameon=True)
    plt.savefig(f"plot/time_varying_{exp_type}_{dataset}.pdf", bbox_inches="tight")
    
    
import matplotlib.pyplot as plt
import numpy as np

def visualize_rss_2(x, y1, y2, y3, y4, y5, dataset):
    """
    x: Number of items
    y1: 2-approx. RSS
    y2: Greedy RSS
    y3: (2+e)-approx. RSS
    y4: RR-LP RSS
    y5: DP RSS (New) - List may be shorter than others
    """
    plt.figure(figsize=(10, 6))
    plt.rcParams.update({"font.size": 16})

    # Existing lines
    plt.plot(x, y1, label="2-approx.", marker="o", linestyle="-", linewidth=2.5, markersize=8, color="blue")
    plt.plot(x, y2, label="Greedy", marker="^", linestyle="--", linewidth=2.5, markersize=8, color="orange")
    plt.plot(x, y3, label="(2+ϵ)-approx.", marker="x", linestyle="-.", linewidth=2.5, markersize=10, color="red")
    plt.plot(x, y4, label="RR-LP", marker="s", linestyle=":", linewidth=2.5, markersize=8, color="purple")
    
    # Corrected DP line: Use x[:len(y5)] to align with correct x-values
    plt.plot(x[:len(y5)], y5, label="DP (Exact)", marker="D", linestyle="-", linewidth=2.5, markersize=8, color="green")
    
    plt.xticks(x)
    plt.xlabel("Number of Items")
    plt.ylabel("RSS (MB)")
    plt.grid(True, linestyle="--", alpha=0.7)
    
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, fontsize=14, frameon=True)

    plt.savefig(f"plot/rss_varying_group_{dataset}.pdf", bbox_inches="tight")
    plt.close()
