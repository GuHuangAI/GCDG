import os
import matplotlib.pyplot as plt

def plot_music(music_data, save_path=None):
        """Visualize muisc feature data."""
        figsize = (19.2, 10.8)
        nrows = len(music_data)
        fig, ax = plt.subplots(nrows=nrows, sharex=True, figsize=figsize)
        ax = [ax] if nrows == 1 else ax
        for i, (key, value) in enumerate(music_data.items()):
            ax[i].plot(value)
            ax[i].set_title(key)
        if save_path is None:
            plt.show()  # interactive
        else:
            fig.savefig(save_path)
        plt.close()