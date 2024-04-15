
import glob
import os
from matplotlib import image as img
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import scipy.ndimage.measurements as ndi
import pandas as pd
import plotly.express as px
from scipy.optimize import curve_fit


# Streamlined data processing (mask files for crop, min/max files for intensity normalization, normbox coords)
# Note: Crop image will change contrast when plotting/gif, set_xlim/ylim will not change contrast.

# Generate data:
def generate_data(image_folder_path, data_path, mask_remove_idx_list = [0,1,2], norm_xyd = [470,80,100], image_idx = []):
    """
    Normalization intensity gotten from 100x100 pixel box. Take Î_i = I_i + In_i - In_0, then I`_i = (2Î_i - Î_max - Î_min)/(Î_max - Î_min)
    0: mask_remove_idx_list \n
    1: norm_xyd \n
    2: [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]] \n
    3: [min_intensity, min_intensity_norm, norm_min_intensity, max_intensity, max_intensity_norm, norm_max_intensity] \n
    4: intensity_array \n
    5: normalized_array \n
    6: norm_intensity_array \n
    7: shift_intensity_array \n
    8: shift_normalized_array \n
    9: shift_norm_intensity_array \n
    10: cropped_image_array \n
    """
    
    # Image paths and mask:
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    if len(mask_remove_idx_list)>1:
        min_path = image_path_list[mask_remove_idx_list[1]]
        max_path = image_path_list[mask_remove_idx_list[2]]
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)

    # Get images and intensity:
    cropped_image_array = []
    intensity_array = []
    normalized_array = []
    for path in image_path_list:
        image = img.imread(path)
        cropped_image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]])
        intensity_array.append(ndi.mean(image, mask))
        normalized_array.append(ndi.mean(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]]))
    cropped_image_array = np.array(cropped_image_array)
    intensity_array = np.array(intensity_array)
    normalized_array = np.array(normalized_array)

    # Normalize:
    norm_intensity_array = intensity_array.copy() + normalized_array[0] - normalized_array
    
    # Shift axis:
    if len(mask_remove_idx_list)>1:
        min_image = img.imread(min_path)
        max_image = img.imread(max_path)
        min_intensity = ndi.mean(min_image, mask)
        max_intensity = ndi.mean(max_image, mask)
        min_intensity_norm = ndi.mean(min_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
        max_intensity_norm = ndi.mean(max_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
        norm_min_intensity = min_intensity + normalized_array[0] - min_intensity_norm
        norm_max_intensity = max_intensity + normalized_array[0] - max_intensity_norm
        shift_intensity_array = -(2*intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
        shift_normalized_array = -(2*normalized_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
        shift_norm_intensity_array = -(2*norm_intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    else:
        min_intensity = ""
        max_intensity = ""
        min_intensity_norm = ""
        max_intensity_norm = ""
        norm_min_intensity = ""
        norm_max_intensity = "" 
        shift_intensity_array = np.array([])
        shift_normalized_array = np.array([])
        shift_norm_intensity_array = np.array([])
    
    f = open(data_path, 'wb')
    np.save(f, mask_remove_idx_list) 
    np.save(f, norm_xyd)
    np.save(f, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]])
    np.save(f, [min_intensity, min_intensity_norm, norm_min_intensity, max_intensity, max_intensity_norm, norm_max_intensity])
    np.save(f, intensity_array)
    np.save(f, normalized_array)
    np.save(f, norm_intensity_array)
    np.save(f, shift_intensity_array)
    np.save(f, shift_normalized_array)
    np.save(f, shift_norm_intensity_array)
    np.save(f, cropped_image_array[image_idx])
    f.close()
    return

def generate_manual_data():
    "Not fixed"
    # Plot only result with clocked_dynamics_paper results:
    # x = [1.0095630935127848, 2.0191282139635764, 3.0286933344143665, 3.9851222750352178, 4.994685368548004, 5.95111633610686, 6.9606794296196455, 7.970244550070437, 9.032943823413161, 9.936236584204071, 10.9989358575468, 11.955366825105655, 12.964931945556447, 14.027631218899174, 14.984058132582016, 15.993623253032808, 16.950054220591664, 17.959619341042455, 19.02231861438518, 19.978745528068025, 21.04144480141075, 22.051009921861546, 23.0074408894204, 24.017001955995184, 24.973432923554032, 26.036132196896762, 26.992563164455618, 27.94899007813847, 28.905421045697324, 30.021254471931975, 30.977685439490838, 31.987246506065613, 33.04994577940834, 34.006376746967206, 35.01594186741799, 36.02550698786878, 36.98193390155162, 37.991499022002415, 39.00106414245321, 40.010629262904004, 41.02019032947878, 41.923487144145696, 43.03932462425637, 43.99574748406321, 44.95217845162207, 46.014877724964784, 46.97130869252365, 48.034007965866365, 49.0967072392091, 50]
    # x = np.arange(1,51)
    # y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
    return

# Gifs and images:
def get_images(image_folder_path, mask_remove_idx_list,norm_xyd, crop_bool):

    # Get paths and images:
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    image_array = []
    norm_zero = img.imread(image_path_list[0])
    norm_zero = ndi.mean(norm_zero[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
    for path in image_path_list:
        image = img.imread(path)
        if crop_bool:
            image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]+ norm_zero - ndi.mean(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]]))  
        else:
            image_array.append(image)
    image_array = np.array(image_array) 
    image_array -= np.min(image_array)
    image_array /= np.max(image_array)
    return 

def make_gifs(image_folder_path, plot_path, mask_remove_idx_list = [0], gif_interval = 500, extra_path = "", save_bool= False):
    def update(i):
        im.set_array(image_array[i])
        return im, 

    # Get paths and images:
    image_path_list = glob.glob(image_folder_path)
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)
    image_array = []
    for image_path in image_path_list:
        image = img.imread(image_path)
        image_array.append(image)
    shape = image_array[0].shape

    # Plot:
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/50.0,shape[0]/50.0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    im = ax.imshow(image_array[0], cmap='Greys_r', animated=True) 
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=gif_interval, blit=True, repeat_delay=10,)
    if save_bool:
        animation_fig.save(plot_path + "Gif" + extra_path + ".gif")
        plt.close(fig)
    else:
        plt.show()
    return

def make_gifs_cropped(image_folder_path, plot_path, mask_remove_idx_list = [0], gif_interval = 500, extra_path = "", save_bool= False):
    def update(i):
        im.set_array(image_array[i])
        return im, 
    
    # Create lists of paths:
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)

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

    #Plot:
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/15.0,shape[0]/15.0)
    ax.set_axis_off() 
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    im = ax.imshow(image_array[0], cmap='Greys_r', animated=True)   
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=gif_interval, blit=True, repeat_delay=10,)
    if save_bool:
        animation_fig.save(plot_path + "Gif_cropped" + extra_path + ".gif")
        plt.close(fig)
    else:
        plt.show()
    return

def show_normalization_box(image_folder_path, plot_path, mask_remove_idx_list = [0], norm_xyd = [470,80,100], rows = 5, cols = 5, plot_dict={}, extra_path = "", save_bool = False):
    "Plot the normalization boxes to check for contaminations."
    # Get paths and images:
    image_path_list = glob.glob(image_folder_path)
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)
    image_array = []
    for path in image_path_list:
        image = img.imread(path)
        image_array.append(image)
    image_array = np.array(image_array) 

    # Plot images:
    plot_dict = plot_dict["norm_dict"]
    fig, axs = plt.subplots(rows,cols,constrained_layout=True)
    fig.set_size_inches(cols,rows)
    axs = axs.flatten()
    for i in range(rows*cols):
        if i < len(image_array):
            axs[i].imshow(image_array[i], cmap='Greys_r') 
            axs[i].text(norm_xyd[0],norm_xyd[1],"t="+str(i), fontsize = plot_dict["text_fontsize"])
        if i == len(image_array):
            axs[i].imshow(np.zeros(image_array[i-1].shape, dtype=np.float16), cmap='Greys')
            axs[i].text(norm_xyd[0]+plot_dict["scalebar_text_x"],norm_xyd[1]+plot_dict["scalebar_text_y"], plot_dict["scalebar_text"], fontsize = plot_dict["scalebar_text_fontsize"], c = plot_dict["scalebar_color"]) 
            axs[i].add_patch(Rectangle((norm_xyd[0]+plot_dict["scalebar_x"],norm_xyd[1]+plot_dict["scalebar_y"]), plot_dict["scalebar_w"], plot_dict["scalebar_h"], edgecolor=plot_dict["scalebar_color"], facecolor=plot_dict["scalebar_color"]))    
        axs[i].set_xlim(norm_xyd[0], norm_xyd[0]+norm_xyd[2]) 
        axs[i].set_ylim(norm_xyd[1]+norm_xyd[2], norm_xyd[1])
        axs[i].set_axis_off()
    if save_bool:
        fig.savefig(plot_path + "Norm_box_images" + extra_path + ".png")
        plt.close(fig)
    else:
        plt.show()
    return

def show_images_cropped(image_folder_path, plot_path, mask_remove_idx_list = [0], norm_xyd = [470,80,100], rows = 5, cols = 5, plot_dict={}, annotate={}, extra_path = "", save_bool = False, enhance_contrast=False):
    "Plot the images cropped by mask."
    # Get paths and images:
    mask_remove_idx_list = mask_remove_idx_list.copy()
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    image_array = []
    norm_zero = img.imread(image_path_list[0])
    norm_zero = ndi.mean(norm_zero[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
    for path in image_path_list:
        image = img.imread(path)
        if enhance_contrast:
            image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]] - ndi.mean(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]]))  
        else:
            image_array.append(image)
    image_array = np.array(image_array) 
    image_array -= np.min(image_array)
    image_array /= np.max(image_array)

    intensity_array = []
    for i in image_array:
        intensity_array.append(ndi.mean(i))

    intensity_array = np.array(intensity_array) 

    plt.plot(np.arange(len(intensity_array)), intensity_array)
    plt.show()
    
    print("Wait")

    # Plot images:
    plot_dict = plot_dict["images_dict"]
    fig, axs = plt.subplots(rows,cols,constrained_layout=True)
    fig.set_size_inches(cols,rows)
    axs = axs.flatten()
    text = ""
    for i in range(rows*cols):
        if i < len(image_array):
            axs[i].imshow(image_array[i], cmap='Greys_r', vmin=0.0, vmax=1.0) 
            # print(np.min(image_array), np.max(image_array))
            if i in annotate:
                text = annotate[i]
            if enhance_contrast:
                axs[i].text(plot_dict["text_shift_x"],plot_dict["text_shift_y"],text+str(i), fontsize = plot_dict["text_fontsize"], color=plot_dict["text_color"])
            else:
                axs[i].text(x_bounds[0]+plot_dict["text_shift_x"],y_bounds[0]+plot_dict["text_shift_y"],text+str(i), fontsize = plot_dict["text_fontsize"], color=plot_dict["text_color"])
                axs[i].set_xlim(x_bounds) 
                axs[i].set_ylim(y_bounds[1],y_bounds[0])
        if i == len(image_array):
            axs[i].imshow(np.zeros(image_array[i-1].shape, dtype=np.float16), cmap='Greys')
            if enhance_contrast:
                axs[i].text(plot_dict["scalebar_text_x"],plot_dict["scalebar_text_y"], plot_dict["scalebar_text"], fontsize = plot_dict["scalebar_text_fontsize"], c = plot_dict["scalebar_color"]) 
                axs[i].add_patch(Rectangle((plot_dict["scalebar_x"],plot_dict["scalebar_y"]), plot_dict["scalebar_w"], plot_dict["scalebar_h"], edgecolor=plot_dict["scalebar_color"], facecolor=plot_dict["scalebar_color"]))    
            else:
                axs[i].text(x_bounds[0]+plot_dict["scalebar_text_x"],y_bounds[0]+plot_dict["scalebar_text_y"], plot_dict["scalebar_text"], fontsize = plot_dict["scalebar_text_fontsize"], c = plot_dict["scalebar_color"]) 
                axs[i].add_patch(Rectangle((x_bounds[0]+plot_dict["scalebar_x"],y_bounds[0]+plot_dict["scalebar_y"]), plot_dict["scalebar_w"], plot_dict["scalebar_h"], edgecolor=plot_dict["scalebar_color"], facecolor=plot_dict["scalebar_color"]))    
                axs[i].set_xlim(x_bounds) 
                axs[i].set_ylim(y_bounds[1],y_bounds[0])
        axs[i].set_axis_off()    
    if save_bool:
        fig.savefig(plot_path + "Images" + extra_path + ".png")
        plt.close(fig)
    else:
        plt.show()
    return

def show_one_image(image_folder_path):
    "To find scalebar-length"
    image_path_list = glob.glob(image_folder_path)
    image_array = []
    for image_path in image_path_list:
        image = img.imread(image_path)
        image_array.append(image)
    shape = image_array[0].shape

    # Plot:
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/50.0,shape[0]/50.0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.imshow(image_array[1], cmap='Greys_r') 
    plt.show()
    return

def show_image_dist_multi(data_path_list, plot_path, image_idx=[], extra_path = "", save_bool = False):
    "Plot_dict not added."
    N = len(data_path_list)
    M = len(image_idx)
    # Get data: 
    mask_remove_idx_list_list = []
    norm_xyd_list = []
    bounds_list = []
    min_max_list = [] 
    intensity_array_list = []
    normalized_array_list = []
    norm_intensity_array_list = []
    shift_intensity_array_list = []
    shift_normalized_array_list = []
    shift_norm_intensity_array_list = []
    cropped_image_array_list = []

    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            mask_remove_idx_list_list.append(np.load(f))
            norm_xyd_list.append(np.load(f))
            bounds_list.append(np.load(f))
            min_max_list.append(np.load(f)) 
            intensity_array_list.append(np.load(f))
            normalized_array_list.append(np.load(f))
            norm_intensity_array_list.append(np.load(f))
            shift_intensity_array_list.append(np.load(f))
            shift_normalized_array_list.append(np.load(f))
            shift_norm_intensity_array_list.append(np.load(f))
            cropped_image_array_list.append(np.load(f))
    
    for array in cropped_image_array_list:
        # print(np.min(array), np.max(array))
        array -= np.min(array)
        array /= np.max(array)
        # plt.hist(np.array(array).flatten(), bins=100)
        # plt.show()
    # for array in cropped_image_array_list:
    #     print(np.min(array), np.max(array))

   
        
    summed_array = sum(cropped_image_array_list)/N
    cropped_image_array_list.append(summed_array)
    cropped_image_array_list = np.array(cropped_image_array_list).reshape(M*(N+1),*cropped_image_array_list[0][0].shape)

    fig, axs = plt.subplots(N+1,M,constrained_layout=True)
    fig.set_size_inches(M+0.5,N+1)
    axs = axs.flatten()
    for i in range((N+1)*M):
        axs[i].imshow(cropped_image_array_list[i], cmap='Greys_r', vmin=0.0, vmax=1.0) 
        if i < M:
            axs[i].text(35,-4,"t="+str(image_idx[i]), fontsize = 8.5)
        if i == M*N:
            axs[i].text(-50,60,"Mean", fontsize = 8.5)
        elif i%M==0:
            axs[i].text(-50,60,"Run "+str(int(i/M)), fontsize = 8.5)
        axs[i].set_axis_off()    

    if save_bool:
        fig.savefig(plot_path + "Image_dist" + extra_path + ".png")
        plt.close(fig)
    else:
        plt.show()
    return


# Intensity plot:
def plot_hysteresis_txt(hysteresis_txt, plot_path, extra_path = "", save_bool = False):
    "Not normalized, not so good"
    if hysteresis_txt=="":
        return
    file = open(hysteresis_txt, 'r')
    Lines = file.readlines()
    x_points = []
    y_points = []
    for line in Lines[1:]:
        field, val , file = line.split('\t')
        x_points.append(float(field.strip()))
        y_points.append(float(val.strip()))
    x_points = np.array(x_points)    
    y_points = np.array(y_points)   
    fig, ax = plt.subplots()
    ax.plot(x_points, y_points, label= "Raw hysteresis")
    fig.legend()
    if save_bool:
        dpi = 200
        fig.savefig(plot_path + "Raw_hysteresis_plot" + extra_path + ".png", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_hysteresis_normalized(hysteresis_txt, data_path, plot_path, extra_path = "", save_bool=False):
    "Normalized and shifted"
    if hysteresis_txt=="":
        return
    file = open(hysteresis_txt, 'r')
    Lines = file.readlines()
    x_points = []
    for line in Lines[1:]:
        field, val , file = line.split('\t')
        x_points.append(float(field.strip()))
    x_points = np.array(x_points)       
    
    with open(data_path, 'rb') as f:
        mask_remove_idx_list = np.load(f)
        norm_xyd = np.load(f)
        bounds = np.load(f)
        min_max = np.load(f)
        intensity_array = np.load(f)
        normalized_array = np.load(f)
        norm_intensity_array = np.load(f)
        shift_intensity_array = np.load(f)
        shift_normalized_array = np.load(f)
        shift_norm_intensity_array = np.load(f)
    
    # Plot normalized hysteresis:
    fig0 = plt.figure()
    plt.plot(x_points, norm_intensity_array, label = "Normalized Hysteresis")
    fig0.legend()

    # Plot shifted hysteresis:
    min_intensity = np.min(norm_intensity_array)
    max_intensity = np.max(norm_intensity_array)
    shift_norm_intensity_array = -(2*norm_intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
    fig1 = plt.figure()
    plt.plot(x_points, shift_norm_intensity_array, label = "Shifted Normalized Hysteresis")
    fig1.legend()
    
    if save_bool:
        dpi = 200
        fig0.savefig(plot_path + "Normalized_hysteresis_plot" + extra_path + ".png", dpi=dpi)
        fig1.savefig(plot_path + "Shifted_normalized_hysteresis_plot" + extra_path + ".png", dpi=dpi)
        plt.close(fig0)
        plt.close(fig1)
    else:
        plt.show()


    return

def plot_intensity_single(data_path, plot_path, extra_path = "", save_bool=False):
    "Plot single data."
    # Get data: 
    with open(data_path, 'rb') as f:
        mask_remove_idx_list = np.load(f)
        norm_xyd = np.load(f)
        bounds = np.load(f)
        min_max = np.load(f)
        intensity_array = np.load(f)
        normalized_array = np.load(f)
        norm_intensity_array = np.load(f)
        shift_intensity_array = np.load(f)
        shift_normalized_array = np.load(f)
        shift_norm_intensity_array = np.load(f)

    fig_list = []

    # Plot raw intensity:
    fig0 = plt.figure()
    font_size=28
    font_family = "Times New Roman"
    plt.scatter(x= 0, y = min_max[0], label = "$\\alpha > \\beta$", color = 'b') 
    plt.rc('font', size=font_size, family=font_family)
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = font_family
    # @mpl.rc_context({'lines.linewidth': 3, 'lines.linestyle': '-'})
    t = np.arange(len(intensity_array))
    plt.step(t, intensity_array, label = "ASI Raw Intensity")
    plt.step(t, normalized_array, label = "Normalization intensity")
    if min_max[0]!="":
        print(min_max[0], min_max[3])
        plt.scatter(x= 0, y = min_max[0], label = "$\\alpha > \\beta$", color = 'b') 
        plt.scatter(x= 0, y = min_max[3], label = "Raw max value", color = 'r') 
        plt.scatter(x= 0, y = min_max[1], label = "Min value norm", color = 'b', marker="x") 
        plt.scatter(x= 0, y = min_max[4], label = "Max value norm", color = 'r', marker="x") 
    plt.grid(axis='both', color="grey", linestyle="--", linewidth="0.25")
    fig0.legend()
    fig_list.append(fig0)

    # Normalized: 
    fig1 = plt.figure()
    plt.step(t, norm_intensity_array, label = "Normalized ASI Intensity")
    if min_max[0]!="":
        plt.axhline(y = min_max[2], label = "Normalized min value", color = 'b', linestyle = '--') 
        plt.axhline(y = min_max[5], label = "Normalized max value", color = 'r', linestyle = '--') 
    plt.grid(axis='both', color="grey", linestyle="--", linewidth="0.25")
    fig1.legend()
    fig_list.append(fig1)

    # Plot shifted intensity:
    if shift_intensity_array.size != 0:
        zero_idx = []
        for j in range(len(shift_intensity_array)-1):
            if shift_intensity_array[j]*shift_intensity_array[j+1]<0:
                zero_idx.append(j)
        print("zero magnetization idx: ", zero_idx)
        fig2 = plt.figure()
        plt.step(t, shift_intensity_array, label = "Shifted ASI Intensity")
        plt.step(t, shift_normalized_array, label = "Shifted Normalization intensity")
        plt.step(t, shift_norm_intensity_array, label = "Shifted Normalized ASI Intensity")
        fig2.legend()
        fig_list.append(fig2)

        # Plot only result with clocked_dynamics_paper results:
        x = np.arange(1,51)
        y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
        fig3 = plt.figure()
        plt.step(t, shift_norm_intensity_array, label = "Normalized ASI Intensity")
        plt.step(x,y, label= "Results in Clocked dynamics paper")
        plt.ylim(-1.1,1.1)
        fig3.legend()
        fig_list.append(fig3)
    
    if save_bool:
        dpi = 200
        fig0.savefig(plot_path + "Raw_intensity_plot" + extra_path + ".png", dpi=dpi)
        fig1.savefig(plot_path + "Normalized_plot" + extra_path + ".png", dpi=dpi)
        if shift_intensity_array.size != 0:
            fig2.savefig(plot_path + "Shifted_plot" + extra_path + ".png", dpi=dpi)
            fig3.savefig(plot_path + "Compare_plot" + extra_path + ".png", dpi=dpi)
        for fig in fig_list: plt.close(fig)
    else:
        plt.show()
    return

def plot_intensity_multi(data_path_list, plot_path, extra_path = "", save_bool=False):
    "Plot multiple runs in same plot"
    
    N = len(data_path_list)
    # Get data: 
    mask_remove_idx_list_list = []
    norm_xyd_list = []
    bounds_list = []
    min_max_list = [] 
    intensity_array_list = []
    normalized_array_list = []
    norm_intensity_array_list = []
    shift_intensity_array_list = []
    shift_normalized_array_list = []
    shift_norm_intensity_array_list = []

    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            mask_remove_idx_list_list.append(np.load(f))
            norm_xyd_list.append(np.load(f))
            bounds_list.append(np.load(f))
            min_max_list.append(np.load(f)) 
            intensity_array_list.append(np.load(f))
            normalized_array_list.append(np.load(f))
            norm_intensity_array_list.append(np.load(f))
            shift_intensity_array_list.append(np.load(f))
            shift_normalized_array_list.append(np.load(f))
            shift_norm_intensity_array_list.append(np.load(f))
    
    fig_list = []

    # Plot raw intensity:
    fig0 = plt.figure()
    for i in range(N):
        t = np.arange(len(intensity_array_list[i]))
        plt.step(t, intensity_array_list[i], label = "Run "+str(i)) # ASI Raw Intensity
    fig0.legend()
    fig_list.append(fig0)

    # Plot normalization intensity:
    fig1 = plt.figure()
    for i in range(N):
        t = np.arange(len(intensity_array_list[i]))
        plt.step(t, normalized_array_list[i], label = "Run "+str(i)) # Normalization intensity
    fig1.legend()
    fig_list.append(fig1)

    # Plot normalized intensity:    
    fig2 = plt.figure()
    if len(min_max_list) != 0:
        avg_norm_min_inten = np.mean([i[2] for i in min_max_list])
        avg_norm_max_inten = np.mean([i[5] for i in min_max_list])
        plt.axhline(y = avg_norm_min_inten, label = "Avg Min", color = 'b', linestyle = '--') 
        plt.axhline(y = avg_norm_max_inten, label = "Avg Max", color = 'r', linestyle = '--') 
    for i in range(N):
        t = np.arange(len(intensity_array_list[i]))
        plt.step(t, norm_intensity_array_list[i], label = "Run "+str(i)) # Normalized ASI Intensity
    fig2.legend()
    fig_list.append(fig2)

    # Plot normalized min/max values:
    fig3 = plt.figure()
    for i in range(N):
        if i==0:
            plt.scatter(i, min_max_list[i][2], label = "Min", color = 'b', marker="x")
            plt.scatter(i, min_max_list[i][5], label = "Max", color = 'r', marker="x")
        else:
            plt.scatter(i, min_max_list[i][2], color = 'b', marker="x")
            plt.scatter(i, min_max_list[i][5], color = 'r', marker="x")
    fig3.legend()
    fig_list.append(fig3)

    # Plot shifted raw intensity:
    fig4 = plt.figure()
    for i in range(N):
        if len(shift_intensity_array_list[0])!=0:
            t = np.arange(len(intensity_array_list[i]))
            plt.step(t, shift_intensity_array_list[i], label = "Run "+str(i)) # Shifted Raw Intensity
    fig4.legend()
    fig_list.append(fig4)

    # Plot shifted normalization intensity
    fig5 = plt.figure()
    for i in range(N):
        if len(shift_normalized_array_list[0])!=0:
            t = np.arange(len(intensity_array_list[i]))
            plt.step(t, shift_normalized_array_list[i], label = "Run "+str(i))  # Shifted Normalization intensity
    fig5.legend()
    fig_list.append(fig5)

    # Plot shifted normalized ASI intensity
    fig6 = plt.figure()
    for i in range(N):
        if len(shift_norm_intensity_array_list[0])!=0:
            t = np.arange(len(intensity_array_list[i]))
            plt.step(t, shift_norm_intensity_array_list[i], label = "Run "+str(i))  #Shifted Normalized ASI Intensity
    fig6.legend()
    fig_list.append(fig6)

    # Plot only result with clocked_dynamics_paper results:
    x = np.arange(1,51)
    y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
    fig7 = plt.figure()
    plt.step(x,y, label= "Ref.")
    for i in range(N):
        if len(shift_norm_intensity_array_list[0])!=0:
            plt.step(t, shift_norm_intensity_array_list[i], label = "Run "+str(i))
    plt.ylim(-1.1,1.1)
    fig7.legend()
    fig_list.append(fig7)

    # Plot mean result with error:
    t = np.arange(len(intensity_array_list[i]))
    mean = np.mean(shift_norm_intensity_array_list, axis=0)
    std = np.std(shift_norm_intensity_array_list, axis=0)
    # print(mean)
    # print(std)
    fig8 = plt.figure()
    plt.scatter(x,y, label= "Ref.", marker="D", s=2)
    plt.errorbar(t, mean, std, fmt="o", c="black", ms=1, elinewidth=1, label="Mean and std", capsize=1.5, capthick=0.5) # ecolor="gray"
    
    plt.ylim(-1.1,1.1)
    fig8.legend()
    fig_list.append(fig8)

    # plt.errorbar(t, mean, error, color = pinkcolors[3], marker='o', linestyle="", ms=marker_size, alpha=0.8, mfc='w')

    if save_bool:
        dpi= 200
        fig0.savefig(plot_path + "Raw_intensity_plot" + extra_path + ".png", dpi=dpi)
        fig1.savefig(plot_path + "Normalization_intensity_plot" + extra_path + ".png", dpi=dpi)
        fig2.savefig(plot_path + "Normalized_ASI_intensity_plot" + extra_path + ".png", dpi=dpi)
        fig3.savefig(plot_path + "Normalized_min_max_plot" + extra_path + ".png", dpi=dpi)
        fig4.savefig(plot_path + "Shifted_Raw_Intensity_plot" + extra_path + ".png", dpi=dpi)
        fig5.savefig(plot_path + "Shifted_Normalization_Intensity_plot" + extra_path + ".png", dpi=dpi)
        fig6.savefig(plot_path + "Shifted_Normalized_ASI_Intensity_plot" + extra_path + ".png", dpi=dpi)
        fig7.savefig(plot_path + "Compare_plot" + extra_path + ".png", dpi=dpi)
        fig8.savefig(plot_path + "Compare_mean_plot" + extra_path + ".png", dpi=dpi)
        for fig in fig_list: plt.close(fig)
    else:
        plt.show()
    return

# Info for each run:
def get_info(case,run_path):
    """
    0: image_folder_path \n
    1: data_path \n
    2: plot_path \n
    3: hysteresis_txt \n
    4: gif_interval \n
    5: mask_remove_idx_list \n
    6: norm_xyd \n
    7: rows \n
    8: cols \n
    9: plot_dict \n
    10: annotate \n
    11: image_idx \n
    """
    
    image_folder_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/"
    data_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Data/"
    plot_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/"
    hysteresis_txt = ""
    norm_xyd = [470,80,100]
    plot_dict = {"norm_dict" : {"scalebar_text_x": 24, "scalebar_text_y": 25, "scalebar_color": "black", "scalebar_text": "10 $\mathrm{\mu}$m", "scalebar_text_fontsize": 8.5,
                "scalebar_x": 6, "scalebar_y": 40, "scalebar_w": 86, "scalebar_h": 10, "text_fontsize": 8.5},
                
                "images_dict": {"scalebar_text_x": 29, "scalebar_text_y": 30, "scalebar_color": "black", "scalebar_text": "10 $\mathrm{\mu}$m", "scalebar_text_fontsize": 8.5,
                "scalebar_x": 11, "scalebar_y": 45, "scalebar_w": 86, "scalebar_h": 10, 
                "text_fontsize": 8.5, "text_shift_x": 2, "text_shift_y": 15, "text_color": "w"}}
    
    images_dict_big = {"scalebar_text_x": 53, "scalebar_text_y": 75, "scalebar_color": "black", "scalebar_text": "20 $\mathrm{\mu}$m", "scalebar_text_fontsize": 8.5,
                "scalebar_x": 7, "scalebar_y": 90, "scalebar_w": 192, "scalebar_h": 20, 
                "text_fontsize": 8.5, "text_shift_x": 2, "text_shift_y": 30, "text_color": "w"}
    annotate = {}
    image_idx = [5,15,25,35,45]

    if case==-5:
        name = "23.11.13_samp1/11.8 hyst 22deg"
        hysteresis_txt = image_folder_path + name + "/file name.txt"
        mask_remove_idx_list = [-1]
        gif_interval = 300
        rows, cols = 14, 10 
        run_path = ""
    elif case==-4:
        name = "23.11.13_samp1/11.8 hyst 0deg"
        hysteresis_txt = image_folder_path + name + "/file name.txt"
        mask_remove_idx_list = [-1]
        gif_interval = 300
        rows, cols = 14, 10 
        run_path = ""
    elif case==-3:
        name = "23.11.13_samp1/11.8 AB 32.5"
        mask_remove_idx_list = [0]
        gif_interval = 200
        rows, cols = 13,8 
        annotate = {0: "Initial, t=", 1: "AB, t=", 71: "ab, t="} 
        run_path = ""
    elif case==-2:
        name = "23.11.13_samp1/11.8 AB 32"
        mask_remove_idx_list = [0]
        gif_interval = 300
        rows, cols = 8,7 
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="} 
        run_path = ""
    elif case==-1:
        name = "23.11.13_samp1/11.8 AB 30"
        mask_remove_idx_list = [0]
        gif_interval = 300
        rows, cols = 8,7
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="} 
        run_path = ""
    elif case==0:
        name = "23.11.29_samp1/11.8 AB ab 33"
        mask_remove_idx_list = [0,2,1]
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="} 
    elif case==1: 
        name = "23.11.29_samp1/12.8 aAbB 33"
        mask_remove_idx_list = [0] #Forgot normalization min/max!
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "aAbB, t=", 36: "AaBb, t="} 
        plot_dict["images_dict"] = images_dict_big
    elif case==2:
        name = "23.11.29_samp1/12.8 AB ab 32"
        mask_remove_idx_list = [0,2,1]
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "AB, t=", 36: "ab, t="}
        plot_dict["images_dict"] = images_dict_big
    elif case==3:
        name = "23.11.29_samp1/11.8 A B AB A B 33"
        mask_remove_idx_list = [0,2,1]
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, t=", 1: "A, t=", 11: "B, t=", 21: "AB, t=", 31: "A, t=", 41: "B, t=",} 
    
    image_folder_path += name + run_path + "/*.png"
    data_path += name.replace("/"," ") + run_path.replace("/"," ") + ".npy"
    plot_path += name.replace("/"," ") + run_path + "/"
    return image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate, image_idx


# Fikse i single og multi. Fikse bildeaksetittel, grid
# Flere runs scatterplot?
# Fjerne intensity fra navn (Raw ASI, Norm., Norm. ASI, Shift i navn.)

# Normalisere bilder? 

# Figurer:
# Hysterese for 0 og 22 grader (1 plot, linjer og scatter.)
# En kjøring med all bildebehandling og normalisering.
# Sammenligne 30, 32 og 33?
# Plot med 10 kjøringer på 33 vs paper, mean/err.
# Plassering av vekst (10,20,30,40)
# Histogram streker. 
# Normalisering av bilder. 

def main():
    # Constants:
    run_path = "/Run "
    multi_plot_path = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/"
    enhance_contrast = True

    # Variables:
    gen_data = True
    run_single = True
    save = False
    extra_path = ""
    case, run_num = 0, 0  #2  
    # 23.11.13: -5: 11.8 hyst 22deg (-), -4: 11.8 hyst 0deg (-), -3: 11.8 AB 32.5 (-), -2: 11.8 AB 32 (-), -1: 11.8 AB 30 (-)
    # 23.11.29: 0: 11.8 AB ab 33 (0-9), 1: 12.8 aAbB 33 (0), 2: 12.8 AB ab 32 (0 & 0 J), 3: 11.8 A B AB A B 33 (0)
    
    multi_plot_path += "11.8 AB ab 33 Run 0-9/"
    cases_runs_dict = {0:[i for i in range(10)]} 

    if run_single:
        # Run single:
        info = get_info(case,run_path+str(run_num))
        image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate, image_idx = info
        if gen_data:
            generate_data(image_folder_path=image_folder_path, data_path=data_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd, image_idx=image_idx)

        if not os.path.exists(plot_path) and save:
            os.makedirs(plot_path)

        # make_gifs(image_folder_path=image_folder_path, plot_path=plot_path, mask_remove_idx_list=mask_remove_idx_list, gif_interval=gif_interval, extra_path = extra_path, save_bool=save)
        # make_gifs_cropped(image_folder_path=image_folder_path, plot_path=plot_path, mask_remove_idx_list=mask_remove_idx_list, gif_interval=gif_interval, extra_path = extra_path, save_bool=save)
        # show_normalization_box(image_folder_path=image_folder_path, plot_path=plot_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd, rows=rows, cols=cols, plot_dict=plot_dict, extra_path=extra_path, save_bool=save)
        # show_images_cropped(image_folder_path=image_folder_path, plot_path=plot_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd, rows=rows, cols=cols, plot_dict=plot_dict, annotate=annotate, extra_path = extra_path, save_bool = save, enhance_contrast=enhance_contrast)
        plot_intensity_single(data_path=data_path, plot_path=plot_path, extra_path = extra_path, save_bool=save)
        # plot_hysteresis_txt(hysteresis_txt=hysteresis_txt, plot_path=plot_path, extra_path = extra_path, save_bool=save)
        # plot_hysteresis_normalized(hysteresis_txt=hysteresis_txt, data_path=data_path, plot_path=plot_path, extra_path = extra_path, save_bool=save)
        # show_one_image(image_folder_path=image_folder_path)
    else:
        # Run multiple:
        data_path_list = []
        for key in cases_runs_dict:
            for element in cases_runs_dict[key]:
                info = get_info(key,run_path+str(element))
                if gen_data:
                    image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate, image_idx = info
                    generate_data(image_folder_path=image_folder_path, data_path=data_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd, image_idx=image_idx)
                data_path_list.append(info[1])
                plot_dict = info[9]
                image_idx = info[11]
        
        if not os.path.exists(multi_plot_path) and save:
            os.makedirs(multi_plot_path)
        # plot_intensity_multi(data_path_list=data_path_list, plot_path=multi_plot_path, extra_path=extra_path, save_bool=save)
        show_image_dist_multi(data_path_list=data_path_list, plot_path=multi_plot_path, image_idx=image_idx, extra_path = extra_path, save_bool = save)
    return

main()




def fordypningsemne():
    # Fordypningsemne etch test: 
    def etch_test():
        test = np.array([[160,  169 ],[240,  251 ],[330,  366 ],[400, 325  ],[450, 297  ],[500,  431 ]]).T
        plt.scatter(*test, label = "Tests")
        plt.scatter(240, 218, label = "Dummy")
        plt.scatter(240, 144, label = "LED")
        plt.legend()
        plt.show()
    
    def IV_curve():
        def ideal(I,I0,n):
            "Uses I = I0 (exp(qV/nkT) -1), where Vt= kT/q = 0.025. Solved for V = f(I)."
            Vt=0.025
            return n* Vt * np.log(I/I0 + 1)
        
        def non_ideal(I,I0, Rs, n):
            "Uses I = I0 (exp(q(V-I*Rs)/nkT) -1), where Vt= kT/q = 0.025. Solved for V = f(I)."
            Vt=0.025
            return I*Rs + n* Vt * np.log(I/I0 + 1)

        # Import IV-data:
        df = pd.read_csv("C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektemne/e1_new.csv")
        df = df[["Reading", "Value"]]
        # print(df)
        fig = px.line(df, x="Value", y="Reading", width=1000, height=900)

        # Curve fit:
        func = ideal
        popt, pcov = curve_fit(func, df["Reading"] , df["Value"])
        print("Ideal curve fit (I0, n): ", popt)
        fig.add_scatter(x=func(df["Reading"], *popt), y= df["Reading"], name="Curve fit ideal", line=dict(dash='dot'))

        func = non_ideal
        popt, pcov = curve_fit(func, df["Reading"] , df["Value"])
        print("Non-ideal curve fit (I0, Rs, n): ", popt)
        fig.add_scatter(x=func(df["Reading"], *popt), y= df["Reading"], name="Curve fit non-ideal", line=dict(dash='dot'))

        # Calculate Rs 
        deltaV = df["Value"][390] - df["Value"][350]
        deltaI = df["Reading"][390] - df["Reading"][350]
        Rs = deltaV/deltaI
        print("Rs:",Rs)

        x_values = np.linspace(1.5,4,500)
        fig.add_scatter(x=x_values, y=x_values*(1/Rs)- 0.83, name="1/R", line=dict(dash='dot')) 

        fig.show()

    def show_image():
        
        image = img.imread("C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektemne/1_004.png")    
        shape = image.shape

        real_array = 5/556
        x = [952,834,808,693,463,448,418,391,369,337]
        x = x - np.roll(x, -1, 0)
        print(x)
        y = np.array(x[:-1])*5/556*1000
        print(y)

        # Plot:
        fig, ax = plt.subplots()
        #fig.set_size_inches(shape[1]/50.0,shape[0]/50.0)
        ax.set_axis_off()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.imshow(image, cmap='Greys_r') 
        plt.show()
        
        return

    #etch_test()
    IV_curve()
    #show_image()
    return

# fordypningsemne()