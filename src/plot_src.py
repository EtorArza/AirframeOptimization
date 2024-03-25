from matplotlib import pyplot as plt
import matplotlib.axes._axes
import pandas as pd
import problem_airframes
import numpy as np
import typing
from tqdm import tqdm as tqdm

def plot_venn_diagram(set_list, total, label_list):
    import supervenn
    fig = supervenn.supervenn(set_list, total, set_annotations=label_list, sets_ordering=None, chunks_ordering='occurence', side_plots=False, col_annotations="percentage")
    plt.show()
    
   
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
                feasability_list.append(problem_airframes.constraint_check_hexarotor_0_1(np.array(x)))

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



if __name__ == '__main__':
    plot_progress_one('results/data/airframes_pyopt_4.csv')
    plot_feasability(*_get_feasability_data_airframes_one_run('results/data/airframes_pyopt_4.csv.log'),'results/data/airframes_pyopt_4.csv.log')