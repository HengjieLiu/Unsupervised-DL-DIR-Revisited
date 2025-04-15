
"""

def reorient_to_LPS(volume, orientation, DEBUG=False)
def get_axi_slice_lps(volume, idx_slice=None)
def get_sag_slice_lps(volume, idx_slice=None)
def get_cor_slice_lps(volume, idx_slice=None)
def display_3view_LPS(volume, orientation='LPS', cmap='gray', vmin=None, vmax=None)
def display_3view_LPS_interactive(volume, orientation='LPS', cmap='gray', vmin=None, vmax=None)
def display_3view_LPS_interactive_v2(volume, orientation='LPS', cmap='gray', vmin=None, vmax=None)
def display_3view_LPS_interactive_v3(volume, orientation='LPS', cmap='gray', vmin=None, vmax=None)
def display_3view_LPS_interactive_alt(volume, orientation='LPS', cmap='gray', vmin=None, vmax=None)
def display_3view_LPS_interactive_dvfgrid_v2(
    dvf_grid_axi, 
    dvf_grid_sag, 
    dvf_grid_cor, 
    orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,)
def plot_3view_features(
    tensor_list, figsize=3, titles=None, suptitle=None, 
    suptitle_fontsize=16, title_fontsize=12,
    cmap='gray', vmin=None, vmax=None, 
    fix_orient=False, orientation='LPS',)
"""



import numpy as np
import matplotlib.pyplot as plt

from utils_basics import *


def reorient_to_LPS(volume, orientation, DEBUG=False):
    """
    Reorient the volume to LPS orientation.

    Parameters:
    - volume: 3D numpy array
    - orientation: 3-letter string, e.g., 'RAS', 'LPS'
    - DEBUG: Boolean, if True, prints operations used on the image volume

    Returns:
    - LPS-oriented 3D numpy array
    """
    orientation = orientation.upper()
    if len(orientation) != 3 or not all(c in 'RLAPSI' for c in orientation):
        raise ValueError("Orientation must be a 3-letter string containing only R, L, A, P, S, I.")
        
        
    ### Gather all information for permutation and flip
    # Define mappings for each axis
    lr_options = {'L', 'R'}
    ap_options = {'A', 'P'}
    si_options = {'S', 'I'}
    
    positive_options = {'L', 'P', 'S'}
    
    # Initialize index (to build permutation)
    lr_index = ap_index = si_index = None
    # Initialize flip axes
    flip_axes = []
    
    # Find the index of each axis in the given string
    for i, char in enumerate(orientation):
        if char in lr_options:
            lr_index = i
            if char not in positive_options:
                flip_axes.append(0)
        elif char in ap_options:
            ap_index = i
            if char not in positive_options:
                flip_axes.append(1)
        elif char in si_options:
            si_index = i
            if char not in positive_options:
                flip_axes.append(2)
    
    permutation = [lr_index, ap_index, si_index]
    flip_axes = sorted(flip_axes)
    
    if DEBUG:
        print('orientation: ', orientation)
        print('permutation: ', permutation)
        print('flip_axes: ', flip_axes)

    
    ### step 1, get permutation to permute to R/L, A/P, S/I
    if permutation != [0, 1, 2]:
        volume = np.transpose(volume, axes=permutation)
        if DEBUG:
            print(f"Applied np.transpose with permutation: {permutation}")
    else:
        if DEBUG:
            print(f"(Dummy) No permutation needed")
    
    ### step 2, flip to LPS
    if flip_axes:
        for axis in flip_axes:
            volume = np.flip(volume, axis=axis)
            if DEBUG:
                print(f"Applied np.flip on axis: {axis}")
    else:
        if DEBUG:
            print(f"(Dummy) No flip needed")
    
    return volume
    
    
def get_axi_slice_lps(volume, idx_slice=None):
    axis = 2
    if idx_slice is None:
        idx_slice = volume.shape[axis]//2
    
    # y: R->L, x: A->P
    img = volume[:,:,idx_slice]
    # y: A->P, x: R->L
    img = np.transpose(img, [1,0])
    
    origin = 'upper'
    
    return img, origin

def get_sag_slice_lps(volume, idx_slice=None):
    axis = 0
    if idx_slice is None:
        idx_slice = volume.shape[axis]//2
    
    # y: A->P, x: I->S
    img = volume[idx_slice,:,:]
    # y: I->S, x: A->P
    img = np.transpose(img, [1,0])
    
    origin = 'lower'
    
    return img, origin
    
def get_cor_slice_lps(volume, idx_slice=None):
    axis = 1
    if idx_slice is None:
        idx_slice = volume.shape[axis]//2
    
    # y: R->L, x: I->S
    img = volume[:,idx_slice,:]
    # y: I->S, x: R->L
    img = np.transpose(img, [1,0])
    
    origin = 'lower'
    
    return img, origin

    
def display_3view_LPS(
    volume, orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,
    ):
    
    ### reorient to LPS
    orientation = orientation.upper()
    if orientation != 'LPS':
        volume = reorient_to_LPS(volume, orientation)
    
    img_axi, origin_axi = get_axi_slice_lps(volume)
    img_sag, origin_sag = get_sag_slice_lps(volume)
    img_cor, origin_cor = get_cor_slice_lps(volume)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title('Axial View')
    axes[0].set_xlabel('R --> L')
    # axes[0].set_ylabel('A --> P')
    axes[0].set_ylabel('P <-- A') # origin is upper
    
    axes[1].imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title('Sagittal View')
    axes[1].set_xlabel('A --> P')
    axes[1].set_ylabel('I --> S')
    
    axes[2].imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title('Coronal View')
    axes[2].set_xlabel('R --> L')
    axes[2].set_ylabel('I --> S')
    
    
    
        
    
import ipywidgets as widgets
from IPython.display import display, clear_output
    
def display_3view_LPS_interactive(
    volume, orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,
    ):
    """
    Interactive 3-view display of a 3D medical image in LPS orientation using sliders.

    Parameters:
    - volume: 3D numpy array
    - orientation: 3-letter string indicating the orientation, default 'LPS'
    - cmap: Colormap for the images, default 'gray'
    - vmin: Minimum intensity for display, default None
    - vmax: Maximum intensity for display, default None
    """
    # Enable interactive matplotlib backend
    # try:
    #     get_ipython().run_line_magic('matplotlib', 'widget')
    # except Exception:
    #     pass  # Magic command may not be available
    
    ### Reorient to LPS if necessary
    orientation = orientation.upper()
    if orientation != 'LPS':
        volume = reorient_to_LPS(volume, orientation)
    
    # Determine the number of slices for each view
    num_axi = volume.shape[2]
    num_sag = volume.shape[0]
    num_cor = volume.shape[1]
    
    # Initialize slice indices to the middle slices
    init_axi = num_axi // 2 + 1
    init_sag = num_sag // 2 + 1
    init_cor = num_cor // 2 + 1
    
    # Create sliders
    axi_slider = widgets.IntSlider(
        min=1, max=num_axi, step=1, value=init_axi,
        description='Axial Slice (I->S):',
        continuous_update=False,
        layout=widgets.Layout(width='500px'),  # Increase the overall slider width
        style={'description_width': '150px'}   # Allocate more space for the description
    )
    sag_slider = widgets.IntSlider(
        min=1, max=num_sag, step=1, value=init_sag,
        description='Sagittal (R->L):',
        continuous_update=False,
        layout=widgets.Layout(width='500px'),  # Increase the overall slider width
        style={'description_width': '150px'}   # Allocate more space for the description
    )
    cor_slider = widgets.IntSlider(
        min=1, max=num_cor, step=1, value=init_cor,
        description='Coronal (A->P):',
        continuous_update=False,
        layout=widgets.Layout(width='500px'),  # Increase the overall slider width
        style={'description_width': '150px'}   # Allocate more space for the description
    )
    
    # axi_slider = widgets.IntSlider(
    #     min=1, max=num_axi, step=1, value=init_axi,
    #     description='Axial (I->S):',
    #     continuous_update=False,
    #     layout=widgets.Layout(width='30%')
    # )
    # sag_slider = widgets.IntSlider(
    #     min=1, max=num_sag, step=1, value=init_sag,
    #     description='Sagittal (R->L):',
    #     continuous_update=False,
    #     layout=widgets.Layout(width='30%')
    # )
    # cor_slider = widgets.IntSlider(
    #     min=1, max=num_cor, step=1, value=init_cor,
    #     description='Coronal (A->P):',
    #     continuous_update=False,
    #     layout=widgets.Layout(width='30%')
    # )
    
    # Create labels to show "current/total"
    axi_label = widgets.Label(value=f"{init_axi}/{num_axi}")
    sag_label = widgets.Label(value=f"{init_sag}/{num_sag}")
    cor_label = widgets.Label(value=f"{init_cor}/{num_cor}")
    
    # Arrange sliders and labels
    axi_box = widgets.HBox([axi_slider, axi_label])
    sag_box = widgets.HBox([sag_slider, sag_label])
    cor_box = widgets.HBox([cor_slider, cor_label])
    
    # Create the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Close the figure to prevent Jupyter from displaying it automatically
    # plt.close(fig)
    
    # Initial display
    img_axi, origin_axi = get_axi_slice_lps(volume, init_axi-1)
    img_sag, origin_sag = get_sag_slice_lps(volume, init_sag-1)
    img_cor, origin_cor = get_cor_slice_lps(volume, init_cor-1)
    
    im_axi = axes[0].imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
    # axes[0].set_title('Axial View')
    axes[0].set_title(f'Axial View (Slice {axi_slider.value}/{num_axi})')
    axes[0].set_xlabel('R --> L')
    axes[0].set_ylabel('P <-- A') # origin is upper
    
    im_sag = axes[1].imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
    # axes[1].set_title('Sagittal View')
    axes[1].set_title(f'Sagittal View (Slice {sag_slider.value}/{num_sag})')
    axes[1].set_xlabel('A --> P')
    axes[1].set_ylabel('I --> S')
    
    im_cor = axes[2].imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
    # axes[2].set_title('Coronal View')
    axes[2].set_title(f'Coronal View (Slice {cor_slider.value}/{num_cor})')
    axes[2].set_xlabel('R --> L')
    axes[2].set_ylabel('I --> S')
    
    plt.tight_layout()
    
    def update_views(change):
        # Update labels
        axi_label.value = f"{axi_slider.value}/{num_axi}"
        sag_label.value = f"{sag_slider.value}/{num_sag}"
        cor_label.value = f"{cor_slider.value}/{num_cor}"
        
        # Update axial view
        img_axi, _ = get_axi_slice_lps(volume, axi_slider.value-1)
        im_axi.set_data(img_axi)
        axes[0].set_title(f'Axial View (Slice {axi_slider.value}/{num_axi})')
        
        # Update sagittal view
        img_sag, _ = get_sag_slice_lps(volume, sag_slider.value-1)
        im_sag.set_data(img_sag)
        axes[1].set_title(f'Sagittal View (Slice {sag_slider.value}/{num_sag})')
        
        # Update coronal view
        img_cor, _ = get_cor_slice_lps(volume, cor_slider.value-1)
        im_cor.set_data(img_cor)
        axes[2].set_title(f'Coronal View (Slice {cor_slider.value}/{num_cor})')
        
        # Redraw the figure
        fig.canvas.draw_idle()
    
    # Attach the update function to sliders
    axi_slider.observe(update_views, names='value')
    sag_slider.observe(update_views, names='value')
    cor_slider.observe(update_views, names='value')
    
    # Display widgets and figure
    ui = widgets.VBox([axi_box, sag_box, cor_box])
    # display(ui, fig.canvas)
    display(ui)

    

def display_3view_LPS_interactive_v2(
    volume, orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,
    ):
    """
    Interactive 3-view display of a 3D medical image in LPS orientation using sliders.

    Parameters:
    - volume: 3D numpy array
    - orientation: 3-letter string indicating the orientation, default 'LPS'
    - cmap: Colormap for the images, default 'gray'
    - vmin: Minimum intensity for display, default None
    - vmax: Maximum intensity for display, default None
    """

    ### Reorient to LPS if necessary
    orientation = orientation.upper()
    if orientation != 'LPS':
        volume = reorient_to_LPS(volume, orientation)
    
    # Determine the number of slices for each view
    num_axi = volume.shape[2]
    num_sag = volume.shape[0]
    num_cor = volume.shape[1]
    
    # Initialize slice indices to the middle slices
    init_axi = num_axi // 2
    init_sag = num_sag // 2
    init_cor = num_cor // 2
    
    # Create sliders
    axi_slider = widgets.IntSlider(
        min=0, max=num_axi-1, step=1, value=init_axi,
        description='Axial Slice (I->S):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    sag_slider = widgets.IntSlider(
        min=0, max=num_sag-1, step=1, value=init_sag,
        description='Sagittal (R->L):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    cor_slider = widgets.IntSlider(
        min=0, max=num_cor-1, step=1, value=init_cor,
        description='Coronal (A->P):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    
    # Create labels to show "current/total"
    axi_label = widgets.Label(value=f"{init_axi+1}/{num_axi}")
    sag_label = widgets.Label(value=f"{init_sag+1}/{num_sag}")
    cor_label = widgets.Label(value=f"{init_cor+1}/{num_cor}")
    
    # Arrange sliders and labels
    axi_box = widgets.HBox([axi_slider, axi_label])
    sag_box = widgets.HBox([sag_slider, sag_label])
    cor_box = widgets.HBox([cor_slider, cor_label])
    
    # Create the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial display
    img_axi, origin_axi = get_axi_slice_lps(volume, init_axi)
    img_sag, origin_sag = get_sag_slice_lps(volume, init_sag)
    img_cor, origin_cor = get_cor_slice_lps(volume, init_cor)
    
    # Display the initial images and store the image objects
    im_axi = axes[0].imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Axial View (Slice {init_axi+1}/{num_axi})')
    axes[0].set_xlabel('R --> L')
    axes[0].set_ylabel('P <-- A')  # origin is upper
    
    im_sag = axes[1].imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Sagittal View (Slice {init_sag+1}/{num_sag})')
    axes[1].set_xlabel('A --> P')
    axes[1].set_ylabel('I --> S')
    
    im_cor = axes[2].imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Coronal View (Slice {init_cor+1}/{num_cor})')
    axes[2].set_xlabel('R --> L')
    axes[2].set_ylabel('I --> S')
    
    plt.tight_layout()
    
    # Display the figure (with ipympl backend)
    # display(fig.canvas)
    
    def update_views(change):
        # Update labels
        axi_label.value = f"{axi_slider.value+1}/{num_axi}"
        sag_label.value = f"{sag_slider.value+1}/{num_sag}"
        cor_label.value = f"{cor_slider.value+1}/{num_cor}"
        
        # Update axial view
        img_axi, _ = get_axi_slice_lps(volume, axi_slider.value)
        im_axi.set_data(img_axi)
        axes[0].set_title(f'Axial View (Slice {axi_slider.value+1}/{num_axi})')
        
        # Update sagittal view
        img_sag, _ = get_sag_slice_lps(volume, sag_slider.value)
        im_sag.set_data(img_sag)
        axes[1].set_title(f'Sagittal View (Slice {sag_slider.value+1}/{num_sag})')
        
        # Update coronal view
        img_cor, _ = get_cor_slice_lps(volume, cor_slider.value)
        im_cor.set_data(img_cor)
        axes[2].set_title(f'Coronal View (Slice {cor_slider.value+1}/{num_cor})')
        
        # Redraw the figure without flashing
        fig.canvas.draw_idle()
    
    # Attach the update function to sliders
    axi_slider.observe(update_views, names='value')
    sag_slider.observe(update_views, names='value')
    cor_slider.observe(update_views, names='value')
    
    # Display widgets
    ui = widgets.VBox([axi_box, sag_box, cor_box])
    display(ui)
    # display(ui, fig.canvas)

##################################################
### To remove!!!
##################################################
import ipywidgets as widgets
from IPython.display import display
from matplotlib.figure import Figure
# from ipympl import FigureWidget

def display_3view_LPS_interactive_v3(
    volume, orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,
    ):
    """
    Interactive 3-view display of a 3D medical image in LPS orientation using sliders.

    Parameters:
    - volume: 3D numpy array
    - orientation: 3-letter string indicating the orientation, default 'LPS'
    - cmap: Colormap for the images, default 'gray'
    - vmin: Minimum intensity for display, default None
    - vmax: Maximum intensity for display, default None
    """

    # Reorient to LPS if necessary
    orientation = orientation.upper()
    if len(orientation) != 3 or not all(c in 'RLAPSI' for c in orientation):
        raise ValueError("Orientation must be a 3-letter string containing only R, L, A, P, S, I.")

    if orientation != 'LPS':
        volume = reorient_to_LPS(volume, orientation)
    
    # Determine the number of slices for each view
    num_axi = volume.shape[2]
    num_sag = volume.shape[0]
    num_cor = volume.shape[1]
    
    # Initialize slice indices to the middle slices
    init_axi = num_axi // 2
    init_sag = num_sag // 2
    init_cor = num_cor // 2
    
    # Create sliders
    axi_slider = widgets.IntSlider(
        min=0, max=num_axi-1, step=1, value=init_axi,
        description='Axial Slice (I->S):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    sag_slider = widgets.IntSlider(
        min=0, max=num_sag-1, step=1, value=init_sag,
        description='Sagittal (R->L):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    cor_slider = widgets.IntSlider(
        min=0, max=num_cor-1, step=1, value=init_cor,
        description='Coronal (A->P):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    
    # Create labels to show "current/total"
    axi_label = widgets.Label(value=f"{init_axi+1}/{num_axi}")
    sag_label = widgets.Label(value=f"{init_sag+1}/{num_sag}")
    cor_label = widgets.Label(value=f"{init_cor+1}/{num_cor}")
    
    # Arrange sliders and labels
    axi_box = widgets.HBox([axi_slider, axi_label])
    sag_box = widgets.HBox([sag_slider, sag_label])
    cor_box = widgets.HBox([cor_slider, cor_label])
    
    # Create the FigureWidget
    fig = Figure(figsize=(15, 5))
    fig_widget = FigureWidget(fig)
    ax_axi, ax_sag, ax_cor = fig.subplots(1, 3)
    
    # Initial display of slices
    img_axi, origin_axi = get_axi_slice_lps(volume, init_axi)
    img_sag, origin_sag = get_sag_slice_lps(volume, init_sag)
    img_cor, origin_cor = get_cor_slice_lps(volume, init_cor)
    
    # Display the initial images
    im_axi = ax_axi.imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_axi.set_title(f'Axial View (Slice {init_axi+1}/{num_axi})')
    ax_axi.set_xlabel('R --> L')
    ax_axi.set_ylabel('P <-- A')  # origin is upper
    
    im_sag = ax_sag.imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_sag.set_title(f'Sagittal View (Slice {init_sag+1}/{num_sag})')
    ax_sag.set_xlabel('A --> P')
    ax_sag.set_ylabel('I --> S')
    
    im_cor = ax_cor.imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
    ax_cor.set_title(f'Coronal View (Slice {init_cor+1}/{num_cor})')
    ax_cor.set_xlabel('R --> L')
    ax_cor.set_ylabel('I --> S')
    
    fig_widget.tight_layout()
    
    # Define the update function
    def update_views(change):
        # Update labels
        axi_label.value = f"{axi_slider.value+1}/{num_axi}"
        sag_label.value = f"{sag_slider.value+1}/{num_sag}"
        cor_label.value = f"{cor_slider.value+1}/{num_cor}"
        
        # Update axial view
        img_axi, _ = get_axi_slice_lps(volume, axi_slider.value)
        im_axi.set_data(img_axi)
        ax_axi.set_title(f'Axial View (Slice {axi_slider.value+1}/{num_axi})')
        
        # Update sagittal view
        img_sag, _ = get_sag_slice_lps(volume, sag_slider.value)
        im_sag.set_data(img_sag)
        ax_sag.set_title(f'Sagittal View (Slice {sag_slider.value+1}/{num_sag})')
        
        # Update coronal view
        img_cor, _ = get_cor_slice_lps(volume, cor_slider.value)
        im_cor.set_data(img_cor)
        ax_cor.set_title(f'Coronal View (Slice {cor_slider.value+1}/{num_cor})')
    
    # Attach the update function to sliders
    axi_slider.observe(update_views, names='value')
    sag_slider.observe(update_views, names='value')
    cor_slider.observe(update_views, names='value')
    
    # Arrange the UI: sliders on top, figure below
    sliders = widgets.VBox([axi_box, sag_box, cor_box])
    ui = widgets.VBox([sliders, fig_widget])
    
    # Display the combined UI
    display(ui)

    
def display_3view_LPS_interactive_alt(
    volume, orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,
    ):
    """
    Alternative interactive 3-view display using interactive_output.
    """

    # Reorient to LPS if necessary
    orientation = orientation.upper()
    if orientation != 'LPS':
        volume = reorient_to_LPS(volume, orientation)
    
    # Determine the number of slices
    num_axi = volume.shape[2]
    num_sag = volume.shape[0]
    num_cor = volume.shape[1]
    
    # Define sliders
    axi_slider = widgets.IntSlider(min=1, max=num_axi, step=1, value=num_axi//2 +1, description='Axial Slice (I->S):')
    sag_slider = widgets.IntSlider(min=1, max=num_sag, step=1, value=num_sag//2 +1, description='Sagittal (R->L):')
    cor_slider = widgets.IntSlider(min=1, max=num_cor, step=1, value=num_cor//2 +1, description='Coronal (A->P):')
    
    # Define the plotting function
    def plot_views(axi, sag, cor):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        img_axi, origin_axi = get_axi_slice_lps(volume, axi-1)
        img_sag, origin_sag = get_sag_slice_lps(volume, sag-1)
        img_cor, origin_cor = get_cor_slice_lps(volume, cor-1)
        
        axes[0].imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0].set_title(f'Axial View (Slice {axi}/{num_axi})')
        axes[0].set_xlabel('R --> L')
        axes[0].set_ylabel('P <-- A')
        
        axes[1].imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1].set_title(f'Sagittal View (Slice {sag}/{num_sag})')
        axes[1].set_xlabel('A --> P')
        axes[1].set_ylabel('I --> S')
        
        axes[2].imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2].set_title(f'Coronal View (Slice {cor}/{num_cor})')
        axes[2].set_xlabel('R --> L')
        axes[2].set_ylabel('I --> S')
        
        plt.tight_layout()
        plt.show()
    
    # Create interactive output
    out = widgets.interactive_output(plot_views, {'axi': axi_slider, 'sag': sag_slider, 'cor': cor_slider})
    
    # Display widgets and output
    ui = widgets.VBox([axi_slider, sag_slider, cor_slider])
    display(ui, out)
    

    
    
def display_3view_LPS_interactive_dvfgrid_v2(
    dvf_grid_axi, 
    dvf_grid_sag, 
    dvf_grid_cor, 
    orientation='LPS', 
    cmap='gray',
    vmin=None, vmax=None,
    ):
    """
    Interactive 3-view display of a 3D medical image in LPS orientation using sliders.

    Parameters:
    - volume: 3D numpy array
    - orientation: 3-letter string indicating the orientation, default 'LPS'
    - cmap: Colormap for the images, default 'gray'
    - vmin: Minimum intensity for display, default None
    - vmax: Maximum intensity for display, default None
    """

    ### Reorient to LPS if necessary
    orientation = orientation.upper()
    if orientation != 'LPS':
        # volume = reorient_to_LPS(volume, orientation)
        dvf_grid_axi = reorient_to_LPS(dvf_grid_axi, orientation)
        dvf_grid_sag = reorient_to_LPS(dvf_grid_sag, orientation)
        dvf_grid_cor = reorient_to_LPS(dvf_grid_cor, orientation)
    
    # Determine the number of slices for each view
    num_axi = dvf_grid_axi.shape[2]
    num_sag = dvf_grid_axi.shape[0]
    num_cor = dvf_grid_axi.shape[1]
    
    # Initialize slice indices to the middle slices
    init_axi = num_axi // 2
    init_sag = num_sag // 2
    init_cor = num_cor // 2
    
    # Create sliders
    axi_slider = widgets.IntSlider(
        min=0, max=num_axi-1, step=1, value=init_axi,
        description='Axial Slice (I->S):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    sag_slider = widgets.IntSlider(
        min=0, max=num_sag-1, step=1, value=init_sag,
        description='Sagittal (R->L):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    cor_slider = widgets.IntSlider(
        min=0, max=num_cor-1, step=1, value=init_cor,
        description='Coronal (A->P):',
        continuous_update=True,
        layout=widgets.Layout(width='500px'),
        style={'description_width': '150px'}
    )
    
    # Create labels to show "current/total"
    axi_label = widgets.Label(value=f"{init_axi+1}/{num_axi}")
    sag_label = widgets.Label(value=f"{init_sag+1}/{num_sag}")
    cor_label = widgets.Label(value=f"{init_cor+1}/{num_cor}")
    
    # Arrange sliders and labels
    axi_box = widgets.HBox([axi_slider, axi_label])
    sag_box = widgets.HBox([sag_slider, sag_label])
    cor_box = widgets.HBox([cor_slider, cor_label])
    
    # Create the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initial display
    img_axi, origin_axi = get_axi_slice_lps(dvf_grid_axi, init_axi)
    img_sag, origin_sag = get_sag_slice_lps(dvf_grid_sag, init_sag)
    img_cor, origin_cor = get_cor_slice_lps(dvf_grid_cor, init_cor)
    
    # Display the initial images and store the image objects
    im_axi = axes[0].imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[0].set_title(f'Axial View (Slice {init_axi+1}/{num_axi})')
    axes[0].set_xlabel('R --> L')
    axes[0].set_ylabel('P <-- A')  # origin is upper
    
    im_sag = axes[1].imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Sagittal View (Slice {init_sag+1}/{num_sag})')
    axes[1].set_xlabel('A --> P')
    axes[1].set_ylabel('I --> S')
    
    im_cor = axes[2].imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
    axes[2].set_title(f'Coronal View (Slice {init_cor+1}/{num_cor})')
    axes[2].set_xlabel('R --> L')
    axes[2].set_ylabel('I --> S')
    
    plt.tight_layout()
    
    # Display the figure (with ipympl backend)
    # display(fig.canvas)
    
    def update_views(change):
        # Update labels
        axi_label.value = f"{axi_slider.value+1}/{num_axi}"
        sag_label.value = f"{sag_slider.value+1}/{num_sag}"
        cor_label.value = f"{cor_slider.value+1}/{num_cor}"
        
        # Update axial view
        img_axi, _ = get_axi_slice_lps(dvf_grid_axi, axi_slider.value)
        im_axi.set_data(img_axi)
        axes[0].set_title(f'Axial View (Slice {axi_slider.value+1}/{num_axi})')
        
        # Update sagittal view
        img_sag, _ = get_sag_slice_lps(dvf_grid_sag, sag_slider.value)
        im_sag.set_data(img_sag)
        axes[1].set_title(f'Sagittal View (Slice {sag_slider.value+1}/{num_sag})')
        
        # Update coronal view
        img_cor, _ = get_cor_slice_lps(dvf_grid_cor, cor_slider.value)
        im_cor.set_data(img_cor)
        axes[2].set_title(f'Coronal View (Slice {cor_slider.value+1}/{num_cor})')
        
        # Redraw the figure without flashing
        fig.canvas.draw_idle()
    
    # Attach the update function to sliders
    axi_slider.observe(update_views, names='value')
    sag_slider.observe(update_views, names='value')
    cor_slider.observe(update_views, names='value')
    
    # Display widgets
    ui = widgets.VBox([axi_box, sag_box, cor_box])
    display(ui)
    # display(ui, fig.canvas)
    
    

def plot_3view_features(
    tensor_list, figsize=3, titles=None, suptitle=None, 
    suptitle_fontsize=16, title_fontsize=12,
    cmap='gray', vmin=None, vmax=None, 
    fix_orient=False, orientation='LPS'):
    """
    Plots 3xN subplots of axial, sagittal, and coronal slices for a list of tensors.
    
    Parameters:
    - tensor_list: List of 5D tensors to plot
    - figsize: Scaling factor for figure size (default 3)
    - titles: List of titles for each tensor (length matches tensor_list)
    - cmap: Colormap for the images (default 'gray')
    - vmin: Minimum intensity for the color map (default None)
    - vmax: Maximum intensity for the color map (default None)
    - fix_orient: If True, reorient the volume to LPS orientation (default False)
    - orientation: Orientation of the volume (default 'LPS')
    - suptitle: Overall title for the plot (default None)
    - suptitle_fontsize: Font size for the suptitle (default 16)
    - title_fontsize: Font size for the subplot titles (default 12)
    """
    
    if titles is None:
        # titles = [''] * len(tensor_list)
        titles = [f'ft_{i+1}' for i in range(len(tensor_list))]
    
    if len(tensor_list) != len(titles):
        raise ValueError("tensor_list and titles should have the same length.")
    
    nrows = 3
    ncols = len(tensor_list)
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize*ncols, figsize*nrows))
    
    for i, tensor in enumerate(tensor_list):
        # Convert the tensor to a NumPy array
        volume = torch2numpy(tensor)
        
        if fix_orient:
            volume = reorient_to_LPS(volume, orientation)
        
        # Get slices for axial, sagittal, and coronal views
        img_axi, origin_axi = get_axi_slice_lps(volume)
        img_sag, origin_sag = get_sag_slice_lps(volume)
        img_cor, origin_cor = get_cor_slice_lps(volume)
        
        # Plot Axial view
        axes[0, i].imshow(img_axi, origin=origin_axi, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[0, i].set_title(f"{titles[i]} (Axial)", fontsize=title_fontsize)
        axes[0, i].set_xlabel('R --> L')
        axes[0, i].set_ylabel('P <-- A')
        axes[0, i].axis('off')  # Turn off axis ticks
        
        # Plot Sagittal view
        axes[1, i].imshow(img_sag, origin=origin_sag, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, i].set_title(f"{titles[i]} (Sagittal)", fontsize=title_fontsize)
        axes[1, i].set_xlabel('A --> P')
        axes[1, i].set_ylabel('I --> S')
        axes[1, i].axis('off')
        
        # Plot Coronal view
        axes[2, i].imshow(img_cor, origin=origin_cor, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[2, i].set_title(f"{titles[i]} (Coronal)", fontsize=title_fontsize)
        axes[2, i].set_xlabel('R --> L')
        axes[2, i].set_ylabel('I --> S')
        axes[2, i].axis('off')
    
    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_fontsize)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to accommodate suptitle
    plt.show()