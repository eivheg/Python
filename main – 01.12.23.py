
import glob
import os
from matplotlib import image as img
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.ndimage.measurements as ndi


# Streamlined data processing (mask files for crop, min/max files for intensity normalization, normbox coords)
# Note: Crop image will change contrast when plotting/gif, set_xlim/ylim will not change contrast.

def make_gifs(image_folder_path, save_path, mask_remove_idx_list = [0], interval = 500, extra_path = "", save_bool= False):
    def update(i):
        im.set_array(image_array[i])
        return im, 
    
    image_path_list = glob.glob(image_folder_path)
    mask_remove_idx_list.sort()
    mask_remove_idx_list.reverse()
    for idx in mask_remove_idx_list:
        image_path_list.pop(idx)
    image_array = []

    for image_path in image_path_list:
        image = img.imread(image_path)
        image_array.append(image)
    shape = image_array[0].shape
    # print(len(image_array))
    # print(shape)
   
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/50.0,shape[0]/50.0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    im = ax.imshow(image_array[0], cmap='Greys_r', animated=True) 
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=interval, blit=True, repeat_delay=10,)
    
    if save_bool:
        animation_fig.save(save_path + extra_path + "gif.gif")
    else:
        plt.show()
    return

def make_gifs_cropped_mask(image_folder_path, save_path, mask_remove_idx_list = [0], interval = 500, extra_path = "", save_bool= False):
    def update(i):
        im.set_array(image_array[i])
        return im, 
    
    # Create lists of paths:
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    mask_remove_idx_list.sort()
    mask_remove_idx_list.reverse()
    for idx in mask_remove_idx_list:
        image_path_list.pop(idx)

    # Create mask points and import images:
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    image_array = []
    for image_path in image_path_list:
        image = img.imread(image_path)
        image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]])
    shape = image_array[0].shape
    # print(len(image_array))
    # print(shape)
   
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/15.0,shape[0]/15.0)
    ax.set_axis_off() 
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    im = ax.imshow(image_array[0], cmap='Greys_r', animated=True)   
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=interval, blit=True, repeat_delay=10,)

    if save_bool:
        animation_fig.save(save_path + "_cropped" + extra_path + ".gif")
    else:
        plt.show()
    return

def plot_hysteresis_txt(txt_path):
    "Not normalized, so not good"
    if txt_path=="":
        return
    file = open(txt_path, 'r')
    Lines = file.readlines()
    x_points = []
    y_points = []
    for line in Lines[1:]:
        field, val , file = line.split('\t')
        x_points.append(float(field.strip()))
        y_points.append(float(val.strip()))
    x_points = np.array(x_points)    
    y_points = np.array(y_points)   

    plt.plot(x_points, y_points)
    plt.show()
    return

def show_normalization_box(image_folder_path, save_path, mask_remove_idx_list = [0], norm_xyd = [470,80,100], rows = 5, cols = 5, text_fontsize= 8.5, extra_path = "", save_bool = False):
    "Plot the normalization boxes to check for contaminations."
    # Sort images and masks/min/max:
    image_path_list = glob.glob(image_folder_path)
    mask_remove_idx_list.sort()
    mask_remove_idx_list.reverse()
    for idx in mask_remove_idx_list:
        image_path_list.pop(idx)

    # Get images:
    image_array = []
    for path in image_path_list:
        image = img.imread(path)
        image_array.append(image)
    image_array = np.array(image_array) 

    # Plot images:
    fig, axs = plt.subplots(rows,cols,constrained_layout=True)
    fig.set_size_inches(cols,rows)
    axs = axs.flatten()
    for i in range(rows*cols):
        if i < len(image_array):
            axs[i].imshow(image_array[i], cmap='Greys_r') 
            axs[i].text(norm_xyd[0],norm_xyd[1],"t="+str(i), fontsize = text_fontsize)
        axs[i].set_xlim(norm_xyd[0], norm_xyd[0]+norm_xyd[2]) 
        axs[i].set_ylim(norm_xyd[1]+norm_xyd[2], norm_xyd[1])
        axs[i].set_axis_off()
    if save_bool:
        fig.savefig(save_path+"_norm_box_images" + extra_path + ".png")
    else:
        plt.show()
    return

def show_images_cropped(image_folder_path, save_path, mask_remove_idx_list = [0], rows = 5, cols = 5, text_fontsize= 8.5, text_shift=(0,0), text_color="w", annotate={}, extra_path = "", save_bool = False, enhance_contrast=False):
    "Plot the images cropped by mask."
    # Sort images and masks/min/max:
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    mask_remove_idx_list.sort()
    mask_remove_idx_list.reverse()
    for idx in mask_remove_idx_list:
        image_path_list.pop(idx)
    
    # Get images and create mask points:
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    image_array = []
    for path in image_path_list:
        image = img.imread(path)
        if enhance_contrast:
            image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]])
        else:
            image_array.append(image)
    image_array = np.array(image_array) 

    # Plot images:
    fig, axs = plt.subplots(rows,cols,constrained_layout=True)
    fig.set_size_inches(cols,rows)
    axs = axs.flatten()
    text = ""
    for i in range(rows*cols):
        if i < len(image_array):
            axs[i].imshow(image_array[i], cmap='Greys_r') 
            if i in annotate:
                text = annotate[i]
            if enhance_contrast:
                axs[i].text(text_shift[0],text_shift[1],text+str(i), fontsize = text_fontsize, color=text_color)
            else:
                axs[i].text(x_bounds[0]+text_shift[0],y_bounds[0]+text_shift[1],text+str(i), fontsize = text_fontsize, color=text_color)
                axs[i].set_xlim(x_bounds) 
                axs[i].set_ylim(y_bounds[1],y_bounds[0])
        axs[i].set_axis_off()    

    if save_bool:
        fig.savefig(save_path + "_images" + extra_path + ".png")
    else:
        plt.show()
    return

def plot_intensity(image_folder_path, save_path, mask_remove_idx_list = [0,1,2], norm_xyd = [470,80,100], save_bool=False):
    "Normalization intensity gotten from 100x100 pixel box. Take Î_i = I_i + In_i - In_0, then I`_i = (2Î_i - Î_max - Î_min)/(Î_max - Î_min)"
    
    # Sort images and masks/min/max:
    image_path_list = glob.glob(image_folder_path)
    if len(mask_remove_idx_list)<3:
        return
    mask_path = image_path_list[mask_remove_idx_list[0]]
    min_path = image_path_list[mask_remove_idx_list[1]]
    max_path = image_path_list[mask_remove_idx_list[2]]
    mask_remove_idx_list.sort()
    mask_remove_idx_list.reverse()
    for idx in mask_remove_idx_list:
        image_path_list.pop(idx)
    
    # Create mask points, get images and raw intensity:
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    min_image = img.imread(min_path)
    max_image = img.imread(max_path)
    min_intensity = ndi.mean(min_image, mask)
    max_intensity = ndi.mean(max_image, mask)
    min_intensity_norm = ndi.mean(min_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
    max_intensity_norm = ndi.mean(max_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
    intensity_array = []
    normalized_array = []
    for path in image_path_list:
        image = img.imread(path)
        intensity_array.append(ndi.mean(image, mask))
        normalized_array.append(ndi.mean(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]]))
    intensity_array = np.array(intensity_array)
    normalized_array = np.array(normalized_array)
    # plt.imshow(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2]+100,norm_xyd[0]:norm_xyd[0]+norm_xyd[2]+100])
    # plt.show()


    # Plot raw intensity:
    fig0 = plt.figure()
    t = np.arange(len(intensity_array))
    plt.step(t, intensity_array, label = "ASI Raw Intensity")
    plt.step(t, normalized_array, label = "Normalization intensity")
    fig0.legend()

    # Normalized: 
    norm_intensity_array = intensity_array.copy() + normalized_array[0] - normalized_array
    min_intensity += normalized_array[0] - min_intensity_norm
    max_intensity += normalized_array[0] - max_intensity_norm
    print("min, mid, max intensity: ", min_intensity, (min_intensity+max_intensity)/2, max_intensity)
    fig1 = plt.figure()
    plt.step(t, norm_intensity_array, label = "Normalized ASI Intensity")
    plt.axhline(y = min_intensity, label = "Min value", color = 'b', linestyle = '--') 
    plt.axhline(y = max_intensity, label = "Max value", color = 'r', linestyle = '--') 
    fig1.legend()

    # Shift axis:
    normalized_array = -(2*normalized_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    intensity_array = -(2*intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    norm_intensity_array = -(2*norm_intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    zero_idx = []
    for j in range(len(intensity_array)-1):
        if intensity_array[j]*intensity_array[j+1]<0:
            zero_idx.append(j)
    print("zero magnetization idx: ", zero_idx)

    # Plot shifted intensity:
    fig2 = plt.figure()
    plt.step(t, intensity_array, label = "ASI Intensity")
    plt.step(t, normalized_array, label = "Normalization intensity")
    plt.step(t, norm_intensity_array, label = "Normalized ASI Intensity")
    #plt.ylim(-1.05,1.05)
    fig2.legend()

    # Plot only result with clocked_dynamics_paper results:
    # x = [1.0095630935127848, 2.0191282139635764, 3.0286933344143665, 3.9851222750352178, 4.994685368548004, 5.95111633610686, 6.9606794296196455, 7.970244550070437, 9.032943823413161, 9.936236584204071, 10.9989358575468, 11.955366825105655, 12.964931945556447, 14.027631218899174, 14.984058132582016, 15.993623253032808, 16.950054220591664, 17.959619341042455, 19.02231861438518, 19.978745528068025, 21.04144480141075, 22.051009921861546, 23.0074408894204, 24.017001955995184, 24.973432923554032, 26.036132196896762, 26.992563164455618, 27.94899007813847, 28.905421045697324, 30.021254471931975, 30.977685439490838, 31.987246506065613, 33.04994577940834, 34.006376746967206, 35.01594186741799, 36.02550698786878, 36.98193390155162, 37.991499022002415, 39.00106414245321, 40.010629262904004, 41.02019032947878, 41.923487144145696, 43.03932462425637, 43.99574748406321, 44.95217845162207, 46.014877724964784, 46.97130869252365, 48.034007965866365, 49.0967072392091, 50]
    x = np.arange(1,51)
    y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
    fig3 = plt.figure()
    plt.step(t, norm_intensity_array, label = "Normalized ASI Intensity")
    plt.step(x,y, label= "Results in Clocked dynamics paper")
    plt.ylim(-1.05,1.05)
    fig3.legend()
    
    if save_bool:
        fig0.savefig(save_path+"_raw_intensity_plot" + extra_path + ".png")
        fig1.savefig(save_path+"_normalized_plot" + extra_path + ".png")
        fig2.savefig(save_path+"_shifted_plot" + extra_path + ".png")
        fig3.savefig(save_path+"_compare_plot" + extra_path + ".png")
    else:
        plt.show()
    return

def plot_notnorm_intensity(image_folder_path, save_path, mask_remove_idx_list = [0,1,2], norm_xyd = [470,80,100], save_bool=False):
    "I`_i = (2Î_i - Î_max - Î_min)/(Î_max - Î_min)"
    
    # Sort images and masks/min/max:
    image_path_list = glob.glob(image_folder_path)
    if len(mask_remove_idx_list)<3:
        return
    mask_path = image_path_list[mask_remove_idx_list[0]]
    min_path = image_path_list[mask_remove_idx_list[1]]
    max_path = image_path_list[mask_remove_idx_list[2]]
    mask_remove_idx_list.sort()
    mask_remove_idx_list.reverse()
    for idx in mask_remove_idx_list:
        image_path_list.pop(idx)
    
    # Create mask points, get images and raw intensity:
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    min_image = img.imread(min_path)
    max_image = img.imread(max_path)
    min_intensity = ndi.mean(min_image, mask)
    max_intensity = ndi.mean(max_image, mask)
    # min_intensity_norm = ndi.mean(min_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
    # max_intensity_norm = ndi.mean(max_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
    intensity_array = []
    # normalized_array = []
    for path in image_path_list:
        image = img.imread(path)
        intensity_array.append(ndi.mean(image, mask))
        # normalized_array.append(ndi.mean(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]]))
    intensity_array = np.array(intensity_array)
    # normalized_array = np.array(normalized_array)

    # Shift axis:
    # normalized_array = -(2*normalized_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    intensity_array = -(2*intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    # norm_intensity_array = -(2*norm_intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    # zero_idx = []
    # for j in range(len(intensity_array)-1):
    #     if intensity_array[j]*intensity_array[j+1]<0:
    #         zero_idx.append(j)
    # print("zero magnetization idx: ", zero_idx)

    # Plot only result with clocked_dynamics_paper results:
    # x = [1.0095630935127848, 2.0191282139635764, 3.0286933344143665, 3.9851222750352178, 4.994685368548004, 5.95111633610686, 6.9606794296196455, 7.970244550070437, 9.032943823413161, 9.936236584204071, 10.9989358575468, 11.955366825105655, 12.964931945556447, 14.027631218899174, 14.984058132582016, 15.993623253032808, 16.950054220591664, 17.959619341042455, 19.02231861438518, 19.978745528068025, 21.04144480141075, 22.051009921861546, 23.0074408894204, 24.017001955995184, 24.973432923554032, 26.036132196896762, 26.992563164455618, 27.94899007813847, 28.905421045697324, 30.021254471931975, 30.977685439490838, 31.987246506065613, 33.04994577940834, 34.006376746967206, 35.01594186741799, 36.02550698786878, 36.98193390155162, 37.991499022002415, 39.00106414245321, 40.010629262904004, 41.02019032947878, 41.923487144145696, 43.03932462425637, 43.99574748406321, 44.95217845162207, 46.014877724964784, 46.97130869252365, 48.034007965866365, 49.0967072392091, 50]
    x = np.arange(1,51)
    y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
    
    fig3 = plt.figure()
    plt.step(np.arange(len(intensity_array)), intensity_array, label = "Normalized ASI Intensity")
    plt.step(x,y, label= "Results in Clocked dynamics paper")
    plt.ylim(-1.05,1.05)
    fig3.legend()
    
    if save_bool:
        fig3.savefig(save_path+"_compare_plot" + extra_path + ".png")
    else:
        plt.show()
    return


def get_path(i):
    """
    0: image_folder_path
    1: save_path
    2: mask_remove_idx_list
    3: interval
    4: hysteresis_txt
    5: norm_xyd
    6: rows
    7: cols 
    8: text_fontsize
    9: text_shift
    10: text_color
    11: annotate
    """
    hysteresis_txt = ""
    norm_xyd = [470,80,100]     #(470,50,100)
    text_fontsize = 8.5
    text_shift = (2,15)
    text_color = "w"
    annotate = {}
    if i==0:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 hyst 0deg/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.13 samp1 11.8 hyst 0deg"
        mask_remove_idx_list = [-1]
        interval = 300
        hysteresis_txt = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 hyst 0deg/file name.txt"
        rows, cols = 13, 11 
    elif i==1:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 ab 33/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.13 samp1 11.8 ab 33"
        mask_remove_idx_list = [0]
        interval = 500
        rows, cols = 8,7 
    elif i==2:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 ab 32.5/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.13 samp1 11.8 ab 32.5"
        mask_remove_idx_list = [0]
        interval = 300
        rows, cols = 13,8
    elif i==3:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/Run 0/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/Run 0"
        mask_remove_idx_list = [0,2,1]
        interval = 300
        rows, cols = 8,7    #13,8
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="} 
    elif i==4:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.29_samp1/11.8 AB ab 33/Run 10/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.29_samp1 11.8 AB ab 33 Run 10"
        mask_remove_idx_list = [0,2,1]
        interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="} 
    elif i==5: #Forgot normalization min/max!
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.29_samp1/12.8 aAbB 33/Run 0/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.29_samp1 12.8 aAbB 33 Run 0"
        mask_remove_idx_list = [0]
        interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "aAbB, t=", 36: "AaBb, t="} 
        text_shift = (2,30)
    elif i==6:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.29_samp1/12.8 AB ab 32/Run 0/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.29_samp1 12.8 AB ab 32 Run 0"
        mask_remove_idx_list = [0,2,1]
        interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="}
        text_shift = (2,30) 
    elif i==7:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.29_samp1/11.8 A B AB A B 33/Run 0/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.29_samp1 11.8 A B AB A B 33 Run 0/"
        mask_remove_idx_list = [0,2,1]
        interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "A, t=", 11: "B, t=", 21: "AB, t=", 31: "A, t=", 41: "B, t=",} 
    elif i==8:
        image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.29_samp1/12.8 AB ab 32/Run 0 J/*.png"
        save_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/23.11.29_samp1 12.8 AB ab 32 Run 0 J"
        mask_remove_idx_list = [0,2,1]
        interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="}
        text_shift = (2,30) 
    return image_folder_path, save_path, mask_remove_idx_list, interval, hysteresis_txt, norm_xyd, rows, cols, text_fontsize, text_shift, text_color, annotate


# 0: image_folder_path
# 1: save_path
# 2: mask_remove_idx_list
# 3: interval
# 4: hysteresis_txt
# 5: norm_xyd
# 6: rows
# 7: cols
# 8: text_fontsize
# 9: text_shift
# 10: text_color
# 11: annotate

n = 7
save = True
extra_path = ""
enhance_contrast = False

if not os.path.exists(get_path(n)[1]) and save:
  os.makedirs(get_path(n)[1])

make_gifs(*get_path(n)[0:4], extra_path = extra_path, save_bool=save)
make_gifs_cropped_mask(*get_path(n)[0:4], extra_path = extra_path, save_bool=save)
# plot_hysteresis_txt(get_path(n)[4])
show_normalization_box(*get_path(n)[0:3], *get_path(n)[5:9], extra_path=extra_path, save_bool=save)
show_images_cropped(*get_path(n)[0:3], *get_path(n)[6:12], extra_path = extra_path, save_bool = save, enhance_contrast=enhance_contrast)
plot_intensity(*get_path(n)[0:3], get_path(n)[5], save_bool=save)
# plot_notnorm_intensity(*get_path(n)[0:3], get_path(n)[5], save_bool=save)


