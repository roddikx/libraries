import numpy as np
import matplotlib.pyplot as plt


def plot_u(u, title=None, save=False, name='newname.jpg', return_fig=False, vmin=None, vmax=None, normalize=False):
    '''u is a 4-D array, and it plots it as a table of plt.imshow(u[i,j]) subplots'''
    
    if vmax in {None} and normalize in {True}: vmax=np.max(u)
    if vmin in {None} and normalize in {True}: vmin=np.min(u)
    
    M1, M2 = u.shape[0], u.shape[1]
    
    # Create a figure with a grid of subplots
    fig, axes = plt.subplots(M1, M2, figsize=(2*M2, 2*M1) )

    # If M1 or M2 is 1, axes may not be a 2D array; ensure it is
    if M1 == 1: axes = np.expand_dims(axes, axis=0)
    if M2 == 1: axes = np.expand_dims(axes, axis=1)
    
    # Loop over each image and display it
    for i in range(M1):
        for j in range(M2):
            axes[i, j].imshow(u[i, j], cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, j].axis('off')  # Optional: turn off axis labels
    
            
    plt.suptitle(title)
    if save==True: plt.savefig(name, dpi=200)
    plt.tight_layout() # Adjust layout and display the plot (optional)
    
    if return_fig: return fig
    else: plt.show()

def save_u_asgif(*arrays, titles=None, filename='animation.gif'):
    
    from imageio import v2 as imageio
    import io
    
    ver, hor = arrays[0].shape[:2]
    
    images = []
    if titles is None: titles = len(arrays) * [''] # if titles is None, I assign all empty titles
    for u, title in zip(arrays, titles):
        fig, axs = plt.subplots(ver, hor)
        if ver == 1: axs = axs.reshape(1,hor)
        if hor == 1: axs = axs.reshape(ver,1)
        for i in range(ver):
            for j in range(hor):
                axs[i,j].imshow(u[i,j], cmap='gray')
                axs[i,j].set_xticks([])
                axs[i,j].set_yticks([])
        plt.suptitle(title)
        # Save the figure to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        # Read the image from the buffer
        image = imageio.imread(buf)
        images.append(image)
        buf.close()
        plt.close(fig)  # Close the figure to free up memory
    
    # Create a GIF from the images
    imageio.mimsave(filename, images, duration=0.5)

def savecurve_asgif(*data, start=0, stop=None, interval=400, filename='curve.gif'):
    import matplotlib.animation as animation

    data = list(data) # data[i] is meant to be a curve (TxNxN shape)
    n = len(data)
    
    for k, x in enumerate(data):
        if isinstance(x, dict):
            first_key = list(x.keys())[0]
            data[k] = x[first_key]
    
    if stop == None: stop = len(data[0])
    
    fig, axes = plt.subplots(1, n, figsize=(1.5*n, 1.5))
    if n == 1: axes = [axes]
    list_of_AxesImage = []
    for k, ax in enumerate(axes):
        list_of_AxesImage += [ax.imshow(data[k][start], cmap='gray', animated=True)]
    
    def update(frame):
        for k, ax in enumerate(list_of_AxesImage):
            ax.set_array(data[k][frame])
        return list_of_AxesImage
    
    ani = animation.FuncAnimation(fig, update, frames=range(start, stop), interval=interval, blit=True)
    ani.save(filename, writer='pillow')
    plt.close(fig)  # Close the figure to prevent display in notebooks
    print(f"GIF saved as {filename}")
    return

def plotcurve_asgif(*data, start=0, stop=None, filename=None, interval=1000):
    delete = False
    if filename == None:
        import os
        filename, delete = 'temp.gif', True
        
    savecurve_asgif(*data, start=start, stop=stop, interval=interval, filename=filename)
    plot_fromgif(gif_path=filename, window_title='Animated GIF', delay=250)
    
    if delete and os.path.exists(filename):
        os.remove(filename)
        print(f"Deleted file: {filename}")
    plt.close()
    return

def plot_fromgif(gif_path='animation.gif', window_title='Animated GIF', delay=250):
    import tkinter as tk
    from PIL import Image, ImageTk, ImageSequence

    # Initialize the Tkinter root window
    root = tk.Tk()
    root.title(window_title)
    
    try:
        frames = [ImageTk.PhotoImage(img)
                  for img in ImageSequence.Iterator(Image.open(gif_path))] # Load the GIF frames
    except Exception as e:
        print(f"Error loading GIF: {e}")
        root.destroy()  # Destroy the root window before exiting
        return

    # Define the update function to animate the GIF
    def update(ind):
        frame = frames[ind % len(frames)]
        label.configure(image=frame)
        root.after(delay, update, ind+1)

    # Create and pack the label widget
    label = tk.Label(root)
    label.pack()

    # Start the animation
    root.after(0, update, 0)
    root.mainloop()
    return

def printcurve(*data, start=0, stop=None):
    '''prints a curve; it can also handle the output of Chambolle-Pock (which is a dictionary)'''
    n = len(data)
    data = list(data)
    for i, x in enumerate(data):
        if type(x) == dict:
            first_key = list(x.keys())[0]
            data[i] = x[first_key]
    if stop == None: stop = len(data[0])
    for t in range(start,stop):
        fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
        if n == 1: axes = [axes]
        for k, ax in enumerate(axes):
            ax.imshow(data[k][t], cmap='gray')
            ax.text(0.5, 0.5, f"Max\n{np.max(data[k][t]):.2f}", ha='center', va='center', fontsize=12)
        plt.show()
    return
