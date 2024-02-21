from matplotlib import pyplot as plt
import supervenn



def plot_venn_diagram(set_list, total, label_list):
    fig = supervenn.supervenn(set_list, total, set_annotations=label_list, sets_ordering=None, chunks_ordering='occurence', side_plots=False, col_annotations="percentage")
    plt.show()
    
   
