import numpy as np
import matplotlib.pyplot as plt
import os

def plot_loss_contours(loss_data, steps, distance, save_fig=False, save_fig_name='loss_contours.png'):
    perturbation_range = np.round(np.linspace(-0.5*distance, 0.5*distance, 8), 3)

    save_fig_name = os.path.join(save_fig_name)
    fig, ax = plt.subplots(1, 1)
    plt.contourf(np.log(loss_data), levels=50)
    ax.set_title('Loss Contours \n'+ r'$L(\theta + \alpha i + \beta j$)')
    ax.axis('square')
    ax.scatter((steps-1)/2., (steps-1)/2., 20, 'r', '*')
    ax.set_xticks(np.linspace(0, steps, 8, endpoint=True))
    ax.set_xticklabels(perturbation_range)
    ax.set_yticks(np.linspace(0, steps, 8, endpoint=True))
    ax.set_yticklabels(perturbation_range)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    plt.colorbar()
    plt.show()

    if save_fig:
        fig.savefig(save_fig_name, transparent=True, dpi=300)