from matplotlib import pyplot as plt
import matplotlib.axes._axes
import pandas as pd
import numpy as np
import typing
from tqdm import tqdm as tqdm
import itertools

marker_list = ["","o","x","s","d","2","^","*"]
linestyle_list = ["-","--","-.", ":",(0, (3, 5, 1, 5, 1, 5)),(5, (10, 3)), (0, (3, 1, 1, 1))]
color_list = ['#000004', "#414487","#2a788e","#22a884","#7ad151","#fde725"]


def plot_venn_diagram(problem_name, set_list, total, label_list):
    import supervenn
    plt.figure(figsize=(10,3))
    fig = supervenn.supervenn(set_list, total, set_annotations=label_list, sets_ordering=None, chunks_ordering='occurrence', side_plots=False, col_annotations="percentage")
    import os
    dir_path = "results/figures/venn"
    os.makedirs(dir_path, exist_ok=True)
    plt.savefig(dir_path + f"/{problem_name}_venn_constraints.pdf")
    plt.close()
    
   
def plot_progress_one(path: str):
    df = pd.read_csv(path, header=0, sep=';')
    fig, ax = plt.subplots(figsize=(3, 2))
    ax.plot(df['time'] / 3600, df['f_best'])
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("f(x)")
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # Set y-axis tick labels to scientific notation
    fig.tight_layout()
    plt.savefig('results/figures/'+path.split("/")[-1].replace('.csv','')+'.pdf')
    plt.close()






def _get_feasability_data_airframes_one_run(path):
    import problem_airframes
    x_list = []
    f_list = []
    feasability_list = []
    i = 0

    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            i += 1
            if 'n_constraint_evals: ' in line:
                x = eval(line.split("x: ")[-1].strip("\n"))
                f = float(line[line.find(' f: ')+len(' f: '):line.rfind(' t: ')] if ' f: ' in line and ' t: ' in line else None)
                f_list.append(f)
                x_list.append(x)
                feasability_list.append(problem_airframes.constraint_check_welf_hexarotor_0_1(np.array(x)))

    # df = pd.DataFrame({'f':f_list, 'g0':list(zip(*feasability_list))[0],  'g1':list(zip(*feasability_list))[1]}) 
    return f_list, feasability_list

def plot_feasability(f_list, constraint_list, path):
   
    n_constraints = len(constraint_list[0])
    print(n_constraints)

    df = pd.DataFrame({'f':f_list, **{'g'+str(idx):list(zip(*constraint_list))[idx] for idx in range(n_constraints)}}) 

    nboxplots = n_constraints + 1

    fig = plt.figure(figsize=(4, 2))
    gs = fig.add_gridspec(1, nboxplots, hspace=0, wspace=0)
    axes:typing.List[matplotlib.axes._axes.Axes] = gs.subplots(sharey=True)
 
    query0 = '(g0 >= 0)'
    for i in range(1,n_constraints):
        query0 += f' and (g{i}>=0)'
    queries = [query0] + [f'g{i} < 0' for i in range(n_constraints)]
    

    labels = ['Feasible'] + [f"Failed g{i}" for i in range(n_constraints)]
    labels = [el + f'\n{df.query(queries[i]).shape[0] / df.shape[0]:.2f}' for i,el in enumerate(labels)]
    print('Its normal that the percentages dont sum to 1.0, ')
    print(r'p(not-A) + p(not-B) + p(A and B) \neq 0 which means we are counting p(not-A and not-B twice:)')

    for i in range(len(queries)):
        axes[i].violinplot(df.query(queries[i])['f'], showmeans=True, showmedians=True, )
        axes[i].set_xlabel(labels[i])

    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    axes[0].spines['right'].set_visible(False)
    for ax in axes:
        ax.tick_params(bottom=False, labelbottom=False)
        ax.tick_params(left=False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    axes[-1].tick_params(left=False)
    axes[-1].spines['left'].set_visible(False)
    
    ylims = axes[0].get_ylim()
    y_ticks = axes[0].get_yticks()
    y_ticks = [el for el in y_ticks if (el > ylims[0] and el < ylims[1])] 
    for ax in axes:
        for y_tick in y_ticks:
            ax.axhline(y=y_tick, color='grey', linestyle='-', alpha=0.25)
    plt.ylim(ylims)
    plt.tight_layout()
    # plt.show()
    plt.savefig('results/figures/f_by_feasability'+path.split("/")[-1].replace('.txt','')+'.pdf')
    plt.close()



def get_f_curves(problem, algorithm, constraint_method, target_column_name):
    data_path = "results/data/"

    df0: pd.DataFrame = pd.read_csv(data_path + problem + "_" + algorithm + "_" + constraint_method + "_2.csv", sep=";")
    n_datapoints = 20
    x_array = np.linspace(1, df0["n_f_evals"].max(), n_datapoints)

    import glob
    file_pattern = data_path + problem + "_" + algorithm + "_" + constraint_method + "_*.csv"
    file_list = glob.glob(file_pattern)
    y_array = np.zeros((len(file_list), n_datapoints))    

    for file_i, file_path in enumerate(file_list):
        df = pd.read_csv(file_path, sep=";")
        assert df0["n_f_evals"].max() == df["n_f_evals"].max() 
        for x_i, x in enumerate(x_array):
            # Get index of row in df that has the largest n_f_evals value that is still lower than x. 
            index_in_df = np.searchsorted(df["n_f_evals"], x, side='left')
            y_array[file_i, x_i] = df[target_column_name][index_in_df]

    return x_array, np.quantile(y_array, 0.5, axis=0), np.quantile(y_array, 0.25, axis=0), np.quantile(y_array, 0.75, axis=0)


def compare_different_constraint_methods(problem, algorithm, constraint_method_list, target_column_name):
    plt.figure(figsize=(4,3))
    for i, constraint_method in enumerate(constraint_method_list):
        x_array, y_median, y_lower, y_upper = get_f_curves(problem, algorithm, constraint_method, target_column_name)
        plt.plot(x_array, y_median, color=color_list[i], marker=marker_list[i], linestyle=linestyle_list[i], label=constraint_method)
        plt.fill_between(x_array, y_lower, y_upper, color=color_list[i], alpha=0.2)
    plt.xlabel("Evaluations")
    plt.ylabel(target_column_name)
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.05), shadow=False, ncol=1)
    plt.tight_layout()

from typing import Iterable
def boxplots_repeatedly_different_train_seed(result_file_path: str, waypoint_name:str):
    import pandas as pd

    df = _read_and_clean_data_every_evaluation_csv(result_file_path)

    for column_name in ["f","nWaypointsReached/nResets","total_energy/nWaypointsReached"]:
        data_list = []
        label_list = []
        max_epoch_list = sorted(df["max_epochs"].unique().tolist())
        for max_epoch in max_epoch_list:
            data_list.append(df.query(f"max_epochs == {max_epoch}")[column_name])    
            label_list.append(str(max_epoch))
            

        plt.figure(figsize=(4,2.5))
        boxplot = plt.boxplot(data_list, showmeans=True)
        # plt.hlines(59.14, *plt.gca().get_xlim(),colors="blue", linestyles="--", label="best retrain 1440s")
        # plt.hlines(55.88, *plt.gca().get_xlim(), colors="red", linestyles="-.", label="best 360s")
        plt.legend()
        plt.xticks(list(range(1, len(data_list)+1)), label_list)
        plt.xlabel("Traininig time (max epochs)")
        plt.ylabel(column_name)
        plt.title("Hex different train seeds")
        plt.tight_layout()
        plt.savefig(f"results/figures/repeatedly_different_train_seed/hex_repeatedly_{column_name.replace('/','-')}_boxplots_{waypoint_name}.pdf")
        plt.close()



def _read_and_clean_data_every_evaluation_csv(details_every_evaluation_csv):
    df = pd.read_csv(details_every_evaluation_csv, sep=";")

    chosen_rows = (df["nWaypointsReached/nResets"] > 0.1) & (df["total_energy/nWaypointsReached"] > 0.01) & (df["total_energy/nWaypointsReached"] < 30.0)
    
    perc_rows = (1.0 - np.count_nonzero(chosen_rows) / df.shape[0]) * 100.0

    print(f"{perc_rows:.1f}% of solutions were discarded as outliers.")

    df = df[chosen_rows]

    df = df.groupby(['hash', 'seed_train', "max_epochs"]).agg({
        'seed_enjoy': lambda x: x.iloc[0] if len(x) > 1 else x.iloc[0],
        'f': 'mean',
        'nWaypointsReached/nResets': 'mean',
        'total_energy/nWaypointsReached': 'mean',
        'total_energy': 'mean'
    }).reset_index()

    return df


def multiobjective_scatter_by_train_time(details_every_evaluation_csv):

    df = _read_and_clean_data_every_evaluation_csv(details_every_evaluation_csv)

    unique_train_seconds = df['max_epochs'].unique()
    color_map = {value: color for value, color in zip(unique_train_seconds, plt.rcParams['axes.prop_cycle'].by_key()['color'])}
    markers = itertools.cycle(('x', 's', 'v', '^', '<', '>', '8', 'p', '*', 'h', 'H', 'D', 'd'))

    labels_map = {
        721: "Hexarotor 720s",
        720: "Optimized design 720s",
    }

    plt.figure(figsize=(10, 6))
    for train_value in unique_train_seconds:
        subset = df[df['max_epochs'] == train_value]
        plt.scatter(
            x=subset['total_energy/nWaypointsReached'],
            y=subset['nWaypointsReached/nResets'],
            label=labels_map.get(train_value, f'max_epochs = {train_value}'),
            color=color_map[train_value],
            marker=next(markers),
            alpha=0.5
        )
    plt.xlabel('Energy per waypoint reached')
    plt.ylabel('Waypoint reached per reset')
    plt.title('Waypoints reached vs. energy use')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/figures/quad_hex_f_variance/scatter_energy_vs_waypoints_MO.pdf")
    return df


def generate_bokeh_interactive_plot(details_every_evaluation_csv, waypoint_name):

    from bokeh.plotting import figure, output_file, show, save, ColumnDataSource
    from bokeh.models import HoverTool
    import pandas as pd
    from airframes_objective_functions import plot_airframe_to_file_isaacgym
    import pickle

    df = _read_and_clean_data_every_evaluation_csv(details_every_evaluation_csv)
    df = df

    imgs =[]
    x=[]
    y=[]
    desc=[]
    colors = []
    markers = []
    legend_labels = []

    for i in tqdm(range(df.shape[0])):
        id = str(df["hash"][i])+ "_" + str(df["seed_train"][i])+ "_" + str(df["seed_enjoy"][i]) + "_" + waypoint_name
        pars_hash = df["hash"][i]
        if pars_hash == 7399056118471101504:
            colors.append("orange")
            markers.append("square")
            legend_labels.append("standard")
        else:
            colors.append("blue")
            markers.append("x")
            legend_labels.append("optimized")

        data = pickle.load(open(f'cache/airframes_animationdata/{id}_airframeanimationdata.wb', 'rb'))
        pars = data["pars"]
        imagepath = f"cache/bokeh_interactive_plot/{id}.png"
        plot_airframe_to_file_isaacgym(pars, imagepath)
        desc.append(str(id))
        imgs.append(imagepath)
        x.append(df["total_energy/nWaypointsReached"][i])
        y.append(df["nWaypointsReached/nResets"][i])

    output_file("test_bokeh.html")

    source = ColumnDataSource(
        data=dict(
            x=x,
            y=y,
            desc=desc,
            imgs=imgs,
            colors=colors,
            legend_labels=legend_labels,
            markers=markers
        )
    )

    hover = HoverTool(
        tooltips="""
        <div>
            <div>
                <img
                    src="@imgs" height="400" alt="@imgs" width="400"
                    style="float: center; margin: 0px 15px 15px 0px;"
                    border="2"
                ></img>
            </div>
        </div>
        """
    )

    p = figure(min_width=400, min_height=400, tools=[hover], title="Mouse over the dots")
    p.scatter('x', 'y', size=10, source=source, fill_alpha=0, line_color='colors', marker="markers", legend_field='legend_labels')


    p.xaxis.axis_label = 'Energy per waypoint reached'
    p.yaxis.axis_label = 'Waypoint reached per reset'
    p.title.text = f'Waypoints reached vs. energy use ({waypoint_name})'


    # p.legend.title = 'Hexarotor Type'
    p.legend.location = 'top_right'
    p.legend.click_policy = 'hide'

    
    save(p)

    with open("test_bokeh.html", "r") as file:
        html = file.read()
    # Add CSS to center the plot vertically and horizontally
    new_html = """
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Bokeh Plot</title>
        <style>
          html, body {
            height: 100%;
            margin: 0;
            display: flex;
            justify-content: center;
            align-items: center;
          }
          #plot-container {
            padding-top: 4cm;
            padding-bottom: 4cm;
          }
        </style>
        <script type="text/javascript" src="https://cdn.bokeh.org/bokeh/release/bokeh-3.1.1.min.js"></script>
        <script type="text/javascript">
            Bokeh.set_log_level("info");
        </script>
      </head>
      <body>
        <div id="plot-container">
    """ + html.split("<body>")[1]

    with open("test_bokeh.html", "w") as file:
        file.write(new_html)





if __name__ == '__main__':
    # plot_progress_one('results/data/airframes_pyopt_4.csv')
    # plot_feasability(*_get_feasability_data_airframes_one_run('results/data/airframes_pyopt_4.csv.log'),'results/data/airframes_pyopt_4.csv.log')

    import os
    dir_path = "results/figures/compare_constraint_methods/"
    os.makedirs(dir_path, exist_ok=True)
    for problem in ["windflo", "toy"]:
        for column in ["f_best", "n_unfeasible_on_ask"]:
            compare_different_constraint_methods(problem, "nevergrad", ["nn_encoding", 'constant_penalty_no_evaluation', 'algo_specific'],column)
            plt.savefig(f"results/figures/compare_constraint_methods/{problem}_nevergrad_{column}.pdf")
            plt.close()
