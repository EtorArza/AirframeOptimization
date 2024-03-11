from matplotlib import pyplot as plt
import pandas as pd


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

if __name__ == '__main__':
    path='results/data/airframes_pyopt_4.csv'
    plot_progress_one(path)