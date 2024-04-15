
import glob
import os
from matplotlib import image as img
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
import numpy as np
import scipy.ndimage.measurements as ndi
import seaborn as sns
from cycler import cycler

# Streamlined data processing (mask files for crop, min/max files for intensity normalization, normbox coords)

# Generate images (assumes mask exists):
def get_images(image_folder_path, mask_remove_idx_list, correction_xyd):

    # Get paths:
    image_path_list = glob.glob(image_folder_path)
    mask_path = image_path_list[mask_remove_idx_list[0]]
    image_path_list = np.delete(image_path_list, mask_remove_idx_list)

    # Get mask bounds:
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])

    # Get images:
    raw_image_array = []
    corrected_image_array = []
    cropped_corrected_image_array = []
    for path in image_path_list:
        image = img.imread(path)
        correction = ndi.mean(image[correction_xyd[1]:correction_xyd[1]+correction_xyd[2],correction_xyd[0]:correction_xyd[0]+correction_xyd[2]])
        raw_image_array.append(image) 
        corrected_image_array.append(image-correction)
        cropped_corrected_image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]]-correction)  
    image = img.imread(image_path_list[0])
    correction_0 = ndi.mean(image[correction_xyd[1]:correction_xyd[1]+correction_xyd[2],correction_xyd[0]:correction_xyd[0]+correction_xyd[2]])
    raw_image_array = np.array(raw_image_array) 
    corrected_image_array = np.array(corrected_image_array) 
    cropped_corrected_image_array = np.array(cropped_corrected_image_array) 
    return raw_image_array, corrected_image_array, cropped_corrected_image_array, [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]], correction_0

# Make Gifs and png:
def make_gifs(image_array, gif_interval, plot_path, extra_path = "", save_bool= False):
    def update(i):
        im.set_array(image_array[i])
        return im, 

    fig, ax = plt.subplots()
    shape = image_array[0].shape
    inches = 9.5
    inches_factor = inches/np.min(shape)
    fig.set_size_inches(shape[1]*inches_factor,shape[0]*inches_factor)
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
    
def show_correction_box(image_array, correction_xyd, rows, cols, correction_dict, plot_path, extra_path = "", save_bool = False):
    "Plot the correction boxes to check for contaminations."
    with plt.rc_context({'font.family': correction_dict["text_font_family"], 'font.size': 15, 'mathtext.fontset': 'stix', 'mathtext.rm': correction_dict["text_font_family"]}): #correction_dict["text_font_size"]
        fig, axs = plt.subplots(rows,cols,constrained_layout=True)
        inches = 9.5
        inches_factor = inches/np.max((rows,cols))
        fig.set_size_inches(cols*inches_factor,rows*inches_factor)
        axs = axs.flatten()
        for i in range(rows*cols):
            if i < len(image_array):
                axs[i].imshow(image_array[i], cmap='Greys_r') 
                axs[i].text(correction_xyd[0]+correction_dict["text_shift_x"],correction_xyd[1]+correction_dict["text_shift_y"],r"$\mathrm{t=1}$t=1"+str(i), style='italic')
                axs[i].text(correction_xyd[0]+correction_dict["text_shift_x"],correction_xyd[1]+correction_dict["text_shift_y"]+20,"t=1$t=1$"+str(i))
            if i == len(image_array):
                axs[i].imshow(np.zeros(image_array[i-1].shape, dtype=np.float16), cmap='Greys')
                axs[i].text(correction_xyd[0]+correction_dict["scalebar_text_x"],correction_xyd[1]+correction_dict["scalebar_text_y"], correction_dict["scalebar_text"], c = correction_dict["scalebar_color"]) 
                axs[i].add_patch(Rectangle((correction_xyd[0]+correction_dict["scalebar_x"],correction_xyd[1]+correction_dict["scalebar_y"]), correction_dict["scalebar_w"], correction_dict["scalebar_h"], edgecolor=correction_dict["scalebar_color"], facecolor=correction_dict["scalebar_color"]))    
            axs[i].set_xlim(correction_xyd[0], correction_xyd[0]+correction_xyd[2]) 
            axs[i].set_ylim(correction_xyd[1]+correction_xyd[2], correction_xyd[1])
            axs[i].set_axis_off()
        if save_bool:
            fig.savefig(plot_path + "Correction_box_images" + extra_path + ".png")
            plt.close(fig)
        else:
            plt.show()
    return

def show_images_cropped(image_array, rows, cols, images_dict, annotate, plot_path, extra_path = "", save_bool = False):
    "Plot the images cropped by mask."
    with plt.rc_context({'font.family': images_dict["text_font_family"], 'font.size': images_dict["text_font_size"], 'mathtext.fontset': images_dict["text_font_set"], 'mathtext.rm': images_dict["text_font_family"]}): #, 'text.usetex': True
        fig, axs = plt.subplots(rows,cols,constrained_layout=True)
        inches = 9.5
        inches_factor = inches/np.max((rows,cols))
        fig.set_size_inches(cols*inches_factor,rows*inches_factor)
        axs = axs.flatten()
        vmin, vmax = np.min(image_array), np.max(image_array)
        text = "$t$="
        for i in range(rows*cols):
            if i < len(image_array):
                axs[i].imshow(image_array[i], cmap='Greys_r', vmin=vmin, vmax=vmax) 
                if i in annotate:
                    text = annotate[i]
                axs[i].text(images_dict["text_shift_x"],images_dict["text_shift_y"],text+str(i), color=images_dict["text_color"], bbox=dict(facecolor=images_dict["bbox_facecolor"], edgecolor=images_dict["bbox_edgecolor"], pad=images_dict["bbox_pad"], alpha= images_dict["bbox_alpha"]))
            elif i == len(image_array):
                axs[i].imshow(np.zeros(image_array[i-1].shape, dtype=np.float16), cmap='Greys')
                axs[i].text(images_dict["scalebar_text_x"],images_dict["scalebar_text_y"], images_dict["scalebar_text"], c = images_dict["scalebar_color"]) 
                axs[i].add_patch(Rectangle((images_dict["scalebar_x"],images_dict["scalebar_y"]), images_dict["scalebar_w"], images_dict["scalebar_h"], edgecolor=images_dict["scalebar_color"], facecolor=images_dict["scalebar_color"]))    
            axs[i].set_axis_off()    
        if save_bool:
            fig.savefig(plot_path + "Images" + extra_path + ".png")
            plt.close(fig)
        else:
            plt.show()
    return

def show_hyst_cropped(image_array, rows, cols, images_dict, annotate, plot_path, extra_path = "", save_bool = False):
    "Plot the images cropped by mask."
    with plt.rc_context({'font.family': images_dict["text_font_family"], 'font.size': images_dict["text_font_size"], 'mathtext.fontset': images_dict["text_font_set"], 'mathtext.rm': images_dict["text_font_family"]}):
        fig, axs = plt.subplots(rows,cols,constrained_layout=True)
        inches = 4.5 #9.5
        inches_factor = inches/np.max((rows,cols))
        fig.set_size_inches(cols*inches_factor,rows*inches_factor)
        axs = axs.flatten()
        vmin, vmax = np.min(image_array), np.max(image_array)
        for i in range(len(annotate)):
            axs[i].imshow(image_array[annotate[i]], cmap='Greys_r', vmin=vmin, vmax=vmax)
            axs[i].text(images_dict["text_shift_x"],images_dict["text_shift_y"],"$t$="+str(annotate[i]), color=images_dict["text_color"], bbox=dict(facecolor=images_dict["bbox_facecolor"], edgecolor=images_dict["bbox_edgecolor"], pad=images_dict["bbox_pad"], alpha= images_dict["bbox_alpha"]))
            axs[i].set_axis_off()   
        axs[len(annotate)].imshow(np.zeros(image_array[len(annotate)-1].shape, dtype=np.float16), cmap='Greys')
        axs[len(annotate)].text(images_dict["scalebar_text_x"],images_dict["scalebar_text_y"], images_dict["scalebar_text"], c = images_dict["scalebar_color"]) 
        axs[len(annotate)].add_patch(Rectangle((images_dict["scalebar_x"],images_dict["scalebar_y"]), images_dict["scalebar_w"], images_dict["scalebar_h"], edgecolor=images_dict["scalebar_color"], facecolor=images_dict["scalebar_color"]))    
        axs[len(annotate)].set_axis_off()   
        if save_bool:
            fig.savefig(plot_path + "Hyst images" + extra_path + ".png")
            plt.close(fig)
        else:
            plt.show()
    return

def show_one_image(image_array):
    "To find scalebar-length"
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    ax.imshow(image_array[2], cmap='Greys_r') 
    plt.show()
    return

def show_image_dist_multi(image_array_array, image_idx, images_multi_dict, plot_path, extra_path = "", save_bool = False):
    "Distribution of domain growth in many runs. Must be same image size, but should be modified!"
    with plt.rc_context({'font.family': images_multi_dict["text_font_family"], 'font.size': images_multi_dict["text_font_size"]}):
        N, M , h, w = image_array_array.shape
        summed_array = sum(image_array_array)/N
        summed_array = summed_array.reshape(1, *summed_array.shape)
        image_array_array = np.concatenate([image_array_array, summed_array], axis=0).reshape(M*(N+1),h, w)
        N += 1

        fig, axs = plt.subplots(N,M,constrained_layout=True)
        inches = 9.5
        inches_factor = inches/np.max((N,M))
        fig.set_size_inches(M*inches_factor+images_multi_dict["figsize_pad_x"],N*inches_factor)
        axs = axs.flatten()
        vmin, vmax = np.min(image_array_array), np.max(image_array_array)
        for i in range(N*M):
            axs[i].imshow(image_array_array[i], cmap='Greys_r', vmin=vmin, vmax=vmax) 
            if i < M:
                axs[i].text(images_multi_dict["text_horizontal_x"], images_multi_dict["text_horizontal_y"],"$t$="+str(image_idx[i]))
            if i == M*(N-1):
                axs[i].text(images_multi_dict["text_vertical_x"], images_multi_dict["text_vertical_y"],"Mean")
            elif i%M==0:
                axs[i].text(images_multi_dict["text_vertical_x"], images_multi_dict["text_vertical_y"],"Run "+str(int(i/M)))
            axs[i].set_axis_off()    

        if save_bool:
            fig.savefig(plot_path + "Image_dist" + extra_path + ".png")
            plt.close(fig)
        else:
            plt.show()
    return

def show_image_compare_multi(image_array_array, image_idx, names, images_multi_dict, plot_path, extra_path = "", save_bool = False):
    "Distribution of domain growth in many runs."

    with plt.rc_context({'font.family': images_multi_dict["text_font_family"], 'font.size': images_multi_dict["text_font_size"]}):
        N, M = len(image_array_array), len(image_array_array[0])

        fig, axs = plt.subplots(N,M,constrained_layout=True)
        inches = 6 # 9.5
        inches_factor = inches/np.max((N,M))
        fig.set_size_inches(M*inches_factor,N*inches_factor-0.2)
        vmin, vmax = np.inf, -np.inf
        for i in range(N):
            vmin = np.min([np.min(image_array_array[i]), vmin])
            vmax = np.max([np.max(image_array_array[i]), vmax])
        for i in range(N):
            for j in range(M):
                axs[i][j].imshow(image_array_array[i][j], cmap='Greys_r', vmin=vmin, vmax=vmax) 
                if i == 0:
                    axs[i][j].text(images_multi_dict["text_horizontal_x"], images_multi_dict["text_horizontal_y"],"$t$="+str(image_idx[j]))
                if j==0:
                    axs[i][j].text(images_multi_dict["text_vertical_x"], images_multi_dict["text_vertical_y"],names[i])
                axs[i][j].set_axis_off()    

        if save_bool:
            fig.savefig(plot_path + "Image_compare" + extra_path + ".png")
            plt.close(fig)
        else:
            plt.show()
    return



# Generate data:
def generate_manual_data():
    "Not fixed"
    # Plot only result with clocked_dynamics_paper results:
    # x = [1.0095630935127848, 2.0191282139635764, 3.0286933344143665, 3.9851222750352178, 4.994685368548004, 5.95111633610686, 6.9606794296196455, 7.970244550070437, 9.032943823413161, 9.936236584204071, 10.9989358575468, 11.955366825105655, 12.964931945556447, 14.027631218899174, 14.984058132582016, 15.993623253032808, 16.950054220591664, 17.959619341042455, 19.02231861438518, 19.978745528068025, 21.04144480141075, 22.051009921861546, 23.0074408894204, 24.017001955995184, 24.973432923554032, 26.036132196896762, 26.992563164455618, 27.94899007813847, 28.905421045697324, 30.021254471931975, 30.977685439490838, 31.987246506065613, 33.04994577940834, 34.006376746967206, 35.01594186741799, 36.02550698786878, 36.98193390155162, 37.991499022002415, 39.00106414245321, 40.010629262904004, 41.02019032947878, 41.923487144145696, 43.03932462425637, 43.99574748406321, 44.95217845162207, 46.014877724964784, 46.97130869252365, 48.034007965866365, 49.0967072392091, 50]
    # x = np.arange(1,51)
    # y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
    return

def generate_data(image_folder_path, hysteresis_txt, data_path, mask_remove_idx_list, norm_xyd):
    """
    Normalization intensity gotten from 100x100 pixel box. Take I_n[i] = I_raw[i] + I_norm[0] - I_norm[i], then I_shift[i] = (2*I_n[i] - I_n(max) - I_n(min))/(I_n(max) - I_n(min))
    0: mask_remove_idx_list \n
    1: norm_xyd \n
    2: [x_bounds[0], x_bounds[1], y_bounds[0], y_bounds[1]] \n
    3: [min_intensity, min_intensity_norm, norm_min_intensity, max_intensity, max_intensity_norm, norm_max_intensity] \n
    4: intensity_array \n
    5: normalization_array \n
    6: norm_intensity_array \n
    7: shift_intensity_array \n
    8: shift_normalized_array \n
    9: shift_norm_intensity_array \n
    10: hyst_x_array \n
    """

    # Get hysteresis x_values:
    if hysteresis_txt!="":
        file = open(hysteresis_txt, 'r')
        Lines = file.readlines()
        hyst_x_array = []
        for line in Lines[1:]:
            field, val , file = line.split('\t')
            hyst_x_array.append(float(field.strip()))
        hyst_x_array = np.array(hyst_x_array)    
    else:
        hyst_x_array = np.array([])
    
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
    intensity_array = []
    normalization_array = []
    for path in image_path_list:
        image = img.imread(path)
        intensity_array.append(ndi.mean(image, mask))
        normalization_array.append(ndi.mean(image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]]))
    intensity_array = np.array(intensity_array)
    normalization_array = np.array(normalization_array)

    # Normalize:
    norm_intensity_array = intensity_array.copy() + normalization_array[0] - normalization_array
    
    # Shift axis:
    if len(mask_remove_idx_list)>1:
        min_image = img.imread(min_path)
        max_image = img.imread(max_path)
        min_intensity = ndi.mean(min_image, mask)
        max_intensity = ndi.mean(max_image, mask)
        min_intensity_norm = ndi.mean(min_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
        max_intensity_norm = ndi.mean(max_image[norm_xyd[1]:norm_xyd[1]+norm_xyd[2],norm_xyd[0]:norm_xyd[0]+norm_xyd[2]])
        norm_min_intensity = min_intensity + normalization_array[0] - min_intensity_norm
        norm_max_intensity = max_intensity + normalization_array[0] - max_intensity_norm
        shift_intensity_array = -(2*intensity_array- norm_min_intensity - norm_max_intensity)/(norm_max_intensity-norm_min_intensity)
        shift_normalized_array = -(2*normalization_array- norm_min_intensity - norm_max_intensity)/(norm_max_intensity-norm_min_intensity)
        shift_norm_intensity_array = -(2*norm_intensity_array- norm_min_intensity - norm_max_intensity)/(norm_max_intensity-norm_min_intensity)
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
    np.save(f, normalization_array)
    np.save(f, norm_intensity_array)
    np.save(f, shift_intensity_array)
    np.save(f, shift_normalized_array)
    np.save(f, shift_norm_intensity_array)
    np.save(f, hyst_x_array)
    f.close()
    return


# Plot histogram:
def plot_pixel_hist(image_array, norm_0, plot_dict):
    # 0,0
    image_array += norm_0
    images_dict = plot_dict["images_dict"] 
    
    with plt.rc_context({'font.family': images_dict["text_font_family"], 'font.size': images_dict["text_font_size"], 'mathtext.fontset': images_dict["text_font_set"], 'mathtext.rm': images_dict["text_font_family"]}):
        # Test threshold:
        fig, axs = plt.subplots(3,1, sharex=True)
        axs[0].hist(image_array.flatten(), bins=int((np.max(image_array)-np.min(image_array))*320), label="All images") #, histtype="step", density=True
        i = 40
        axs[1].hist(image_array[i].flatten(), bins=int((np.max(image_array[i])-np.min(image_array[i]))*320), label="Image 40") 
        i = 43
        axs[2].hist(image_array[i].flatten(), bins=int((np.max(image_array[i])-np.min(image_array[i]))*320), label="Image 43") 
        fig.supxlabel('Pixel values [unitless]')
        fig.supylabel('Counts [unitless]')
        for ax in axs:
            ax.grid(axis='both', color="grey", linestyle="--", linewidth="0.25")
            ax.legend()
        plt.show()


        image_array = np.where(image_array>0.427, -1, 1) #-0.071
        test = [ndi.mean(i) for i in image_array]

        with open("C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Data/23.11.29_samp1 11.8 AB ab 33 Run 0.npy", 'rb') as f:
            mask_remove_idx_list = np.load(f)
            norm_xyd = np.load(f)
            bounds = np.load(f)
            min_max = np.load(f)
            intensity_array = np.load(f)
            normalization_array = np.load(f)
            norm_intensity_array = np.load(f)
            shift_intensity_array = np.load(f)
            shift_normalized_array = np.load(f)
            shift_norm_intensity_array = np.load(f)
            hyst_x_array = np.load(f)
        
        plt.step(np.arange(len(shift_norm_intensity_array)), shift_norm_intensity_array, c= plot_dict["colors"]["b"], label="Min/max")
        plt.step(np.arange(len(image_array)), test, c= plot_dict["colors"]["r"], label="Threshold")
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M_\mathrm{s}$]')
        plt.grid(axis='both', color="grey", linestyle="--", linewidth="0.25")
        plt.legend()
        plt.show()
        
                
    #18,21,43
        fig, axs = plt.subplots(1,4,constrained_layout=True)
        axs[0].imshow(image_array[18], vmin=-1, vmax=1, cmap="Greys")
        axs[1].imshow(image_array[21], vmin=-1, vmax=1, cmap="Greys")
        axs[2].imshow(image_array[43], vmin=-1, vmax=1, cmap="Greys")
        axs[3].imshow(np.zeros(image_array[18].shape, dtype=np.float16), cmap='Greys')
        for ax in axs:
            ax.set_axis_off()
            if ax != axs[3]:
                autoAxis = ax.axis()
                rec = Rectangle((autoAxis[0]-0.7,autoAxis[2]-0.2),(autoAxis[1]-autoAxis[0])+1,(autoAxis[3]-autoAxis[2])+0.4,fill=False,lw=0.7)
                rec = ax.add_patch(rec)
                rec.set_clip_on(False) 
        
        axs[0].text(2,-5,"$t$=18") 
        axs[1].text(2,-5,"$t$=21")
        axs[2].text(2,-5,"$t$=43")
        axs[3].text(images_dict["scalebar_text_x"],images_dict["scalebar_text_y"], images_dict["scalebar_text"], c = images_dict["scalebar_color"]) 
        axs[3].add_patch(Rectangle((images_dict["scalebar_x"],images_dict["scalebar_y"]), images_dict["scalebar_w"], images_dict["scalebar_h"], edgecolor=images_dict["scalebar_color"], facecolor=images_dict["scalebar_color"]))    
        plt.show()    
    

    # image_array -= np.min(image_array)
    # image_array /= np.max(image_array)
    # print(np.round(np.max(image_array),3), np.round(np.min(image_array),3))
    # print(np.arange(0.3,0.6+0.001,0.001))
    # print(image_array.shape)
    # print( (int((np.max(image_array)-np.min(image_array))*320)) )
    # print((np.max(image_array)-np.min(image_array))/ int((np.max(image_array)-np.min(image_array))*320) )
    # plt.hist(image_array.flatten(), bins= np.arange(np.round(np.min(image_array),3), np.round(np.max(image_array),3)+0.00312,0.00312) )
    
    return

# Intensity plot:
def plot_hysteresis_txt(hysteresis_txt, plot_path, extra_path = "", save_bool = False):
    "Not normalized, not so good, not updated so don't use!"
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
    ax.plot(x_points, y_points, label= "Raw ASI")
    fig.legend()
    if save_bool:
        dpi = 200
        fig.savefig(plot_path + "Raw_hysteresis_plot_txt" + extra_path + ".png", dpi=dpi)
        plt.close(fig)
    else:
        plt.show()
    return

def plot_hysteresis_normalized(data_path, plot_dict, plot_path, extra_path = "", save_bool=False):
    "Plot normalized hysteresis curve."
    # Get data: 
    with open(data_path, 'rb') as f:
        mask_remove_idx_list = np.load(f)
        norm_xyd = np.load(f)
        bounds = np.load(f)
        min_max = np.load(f)
        intensity_array = np.load(f)
        normalization_array = np.load(f)
        norm_intensity_array = np.load(f)
        shift_intensity_array = np.load(f)
        shift_normalized_array = np.load(f)
        shift_norm_intensity_array = np.load(f)
        hyst_x_array = np.load(f)
    fig_list = []

    if hyst_x_array.size == 0:
        return
    
    intensity_dict = plot_dict["intensity_dict"]
    colors = plot_dict["colors"]

    with plt.rc_context({'font.family': intensity_dict["text_font_family"], 'font.size': intensity_dict["text_font_size"], 'mathtext.fontset': intensity_dict["text_font_set"], 'mathtext.rm': intensity_dict["text_font_family"]}):
        # Plot raw intensity:
        fig0 = plt.figure()
        plt.plot(hyst_x_array, intensity_array, label = "Raw ASI", c= colors["i"])
        plt.plot(hyst_x_array, normalization_array, label = "Normalization", c= colors["r"])
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig0.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig0)

        # Plot normalized intensity: 
        fig1 = plt.figure()
        plt.plot(hyst_x_array, norm_intensity_array, label = "Normalized ASI", c= colors["b"])
        plt.grid(axis='both', color="grey", linestyle="--", linewidth="0.25")
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig1.legend(loc='center left', bbox_to_anchor=(0.6, 0.835)) 
        fig_list.append(fig1)

        # Plot shifted intensity:
        min_intensity = np.min(norm_intensity_array)
        max_intensity = np.max(norm_intensity_array)
        shift_norm_intensity_array = -(2*norm_intensity_array- min_intensity - max_intensity)/(max_intensity-min_intensity)
        zero_idx = []
        for j in range(len(shift_norm_intensity_array)-1):
            if shift_norm_intensity_array[j]*shift_norm_intensity_array[j+1]<0:
                zero_idx.append(j)
        print("Zero magnetization idx: ", zero_idx)

        fig2 = plt.figure()
        plt.plot(hyst_x_array, shift_norm_intensity_array, label = "Normalized ASI", c= colors["b"])
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Magnetization [$M/M_\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig2.legend(loc='center left', bbox_to_anchor=(0.123, 0.835)) 
        fig_list.append(fig2)
        
        with open(plot_path + "Data" + extra_path + ".txt", 'w') as f:
            f.write("Cross zero (mT): [" + str(hyst_x_array[zero_idx[0]])+","+str(hyst_x_array[zero_idx[1]])+"] \n")
            f.close()

        if save_bool:
            dpi = 200
            fig0.savefig(plot_path + "Raw_hysteresis_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig1.savefig(plot_path + "Normalized_hysteresis_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig2.savefig(plot_path + "Shifted_hysteresis_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")  
            for fig in fig_list: plt.close(fig)
        else:
            plt.show()
    return

def plot_intensity_single(data_path, plot_dict, plot_path, extra_path = "", save_bool=False):
    "Plot single intensity data, not for hysteresis."
    # Get data: 
    with open(data_path, 'rb') as f:
        mask_remove_idx_list = np.load(f)
        norm_xyd = np.load(f)
        bounds = np.load(f)
        min_max = np.load(f)
        intensity_array = np.load(f)
        normalization_array = np.load(f)
        norm_intensity_array = np.load(f)
        shift_intensity_array = np.load(f)
        shift_normalized_array = np.load(f)
        shift_norm_intensity_array = np.load(f)
        hyst_x_array = np.load(f)

    if hyst_x_array.size > 0:
        return

    intensity_dict = plot_dict["intensity_dict"]
    colors = plot_dict["colors"]

    with plt.rc_context({'font.family': intensity_dict["text_font_family"], 'font.size': intensity_dict["text_font_size"], 'mathtext.fontset': intensity_dict["text_font_set"], 'mathtext.rm': intensity_dict["text_font_family"]}):
        fig_list = []
        t = np.arange(len(intensity_array))

        # Plot raw intensity:
        fig0 = plt.figure()
        plt.step(t, intensity_array, label = "Raw ASI", c= colors["i"]) 
        if min_max[0]!="":
            plt.scatter(x= -1, y = min_max[0], label = "Raw min", color = colors["i"], marker="^") 
            plt.scatter(x= -2, y = min_max[3], label = "Raw max", color = colors["i"], marker="v") 
            plt.scatter(x= -1, y = min_max[1], color = colors["r"], marker="_") 
            plt.scatter(x= -2, y = min_max[4], color = colors["r"], marker="_") 
        plt.step(t, normalization_array, label = "Correction", c= colors["r"])
        plt.xlabel('Time [steps]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig0.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig0)

        # Plot normalized intensity: 
        fig1 = plt.figure()
        plt.step(t, norm_intensity_array, label = "Corrected ASI", c=colors["b"])
        if min_max[0]!="":
            plt.axhline(y = min_max[2], label = "Corrected min", color = 'b', linestyle = '--') 
            plt.axhline(y = min_max[5], label = "Corrected max", color = 'r', linestyle = '--') 
        plt.xlabel('Time [steps]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig1.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig1)

        # Plot shifted intensity:
        zero_idx = []
        if shift_norm_intensity_array.size != 0:
            for j in range(len(shift_norm_intensity_array)-1):
                if shift_norm_intensity_array[j]*shift_norm_intensity_array[j+1]<0:
                    zero_idx.append(j)
            fig2 = plt.figure()
            plt.step(t, shift_intensity_array, label = "Raw ASI", c=colors["i"])
            plt.step(t, shift_normalized_array, label = "Correction", c=colors["r"])
            plt.step(t, shift_norm_intensity_array, label = "Corrected ASI", c=colors["b"])
            plt.ylim(-1.1,1.1)
            plt.xlabel('Time [steps]')
            plt.ylabel('Magnetization [$M/M_\mathrm{s}$]')
            plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
            fig2.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
            fig_list.append(fig2)

            # Plot only result with clocked_dynamics_paper results:
            x = np.arange(1,51)
            y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
            # y_manually = [-1.13517607, -1.09036194, -0.86656981, -0.96472069, -0.85954462, -0.89885679, -0.81527808, -0.93192596, -0.90191934, -0.85617546, -0.81557672, -0.77670768, -0.71112972, -0.54604082, -0.59358175, -0.52068988, -0.494679, -0.41149262, -0.35641125, -0.30297984, -0.23238309, -0.2018896, -0.14167957, -0.11956197, -0.06621322, -0.01502974, 0.06318063, 0.11459734, 0.20283186, 0.25869927, 0.30623321, 0.35354246, 0.40553616, 0.45189753, 0.49120605, 0.3673021, 0.23708137, -0.04039539, -0.02771522, -0.16439639, -0.29125459, -0.42867972, -0.56520658, -0.69583158, -0.80007541, -0.9177695 , -0.96089781, -0.99384266, -1.01948445, -0.99562884]
            # x_manually = np.arange(len(y_manually))
            #print(len(y_manually))
            fig3 = plt.figure()
            # plt.step(x_manually,y_manually, label= "Manual run", c=colors["g"])
            plt.step(t, shift_norm_intensity_array, label = "Corrected ASI", c=colors["b"]) #Corrected ASI Automated run 
            plt.step(x,y, label= "Reference", c=colors["o"])
            plt.ylim(-1.1,1.1)
            plt.xlabel('Time [steps]')
            plt.ylabel('Magnetization [$M/M_\mathrm{s}$]')
            plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
            fig3.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
            fig_list.append(fig3)
        
        if len(zero_idx)>0:
            with open(plot_path + "Data" + extra_path + ".txt", 'w') as f:
                f.write("Cross zero (idx): [" + str(zero_idx[0])+","+str(zero_idx[1])+"] \n")
                f.write("Shifted min/max: " + str(np.min(shift_norm_intensity_array)) + ", " + str(np.max(shift_norm_intensity_array)))
                f.close()

        if save_bool:
            dpi = 200
            fig0.savefig(plot_path + "Raw_intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig1.savefig(plot_path + "Normalized_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            if shift_norm_intensity_array.size != 0:
                fig2.savefig(plot_path + "Magnetization_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
                fig3.savefig(plot_path + "Compare_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            for fig in fig_list: plt.close(fig)
        else:
            plt.show()
    return

def plot_hysteresis_multi(data_path_list, plot_dict, hyst_names, plot_path, extra_path = "", save_bool=False):
    "Plot multiple hysteresis in same plot"
    
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
    hyst_x_array_list = []

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
            hyst_x_array_list.append(np.load(f))
    
    if len(hyst_x_array_list[0])==0:
        return
    
    fig_list = []
    colors = plot_dict["colors"]
    keys = [key for key in colors]
    intensity_dict = plot_dict["intensity_dict"]

    with plt.rc_context({'font.family': intensity_dict["text_font_family"], 'font.size': intensity_dict["text_font_size"], 'mathtext.fontset': intensity_dict["text_font_set"], 'mathtext.rm': intensity_dict["text_font_family"]}):
        # Plot raw intensity:
        fig0 = plt.figure()
        for i in range(N):
            t = hyst_x_array_list[i]
            plt.plot(t, intensity_array_list[i], label = hyst_names[i], c=colors[keys[i]]) # ASI Raw Intensity
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig0.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig0)

        # Plot normalization intensity:
        fig1 = plt.figure()
        for i in range(N):
            t = hyst_x_array_list[i]
            plt.plot(t, normalized_array_list[i], label = hyst_names[i], c=colors[keys[i]]) # Normalization intensity
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig1.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig1)

        # Plot normalized ASI intensity:    
        fig2 = plt.figure()
        for i in range(N):
            t = hyst_x_array_list[i]
            plt.plot(t, norm_intensity_array_list[i], label = hyst_names[i], c=colors[keys[i]]) # Normalized ASI Intensity
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig2.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig2)

        # Plot shifted normalized ASI intensity
        fig3 = plt.figure()
        zero = []
        for i in range(N):
            t = hyst_x_array_list[i]
            min_intensity = np.min(norm_intensity_array_list[i])
            max_intensity = np.max(norm_intensity_array_list[i])
            shift_norm_intensity_array = -(2*norm_intensity_array_list[i]- min_intensity - max_intensity)/(max_intensity-min_intensity)
            zero_idx = []
            for j in range(len(shift_norm_intensity_array)-1):
                if shift_norm_intensity_array[j]*shift_norm_intensity_array[j+1]<0:
                    zero_idx.append(j)
            zero.append(t[zero_idx])
            plt.plot(t, shift_norm_intensity_array, label = hyst_names[i], c=colors[keys[i]])  #Shifted Normalized ASI Intensity
        plt.ylim(-1.1,1.1)
        plt.xlabel('Magnetic field [mT]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig3.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig3)

        with open(plot_path + "Data" + extra_path + ".txt", 'w') as f:
            f.write("Cross zero (mT): " + "".join("[" + str(i[0])+","+str(i[1])+"] " for i in zero) + "\n")
            f.close()

        if save_bool:
            dpi= 200
            fig0.savefig(plot_path + "Raw_hysteresis_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig1.savefig(plot_path + "Normalization_intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig2.savefig(plot_path + "Normalized_hysteresis_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig3.savefig(plot_path + "Shifted_hysteresis_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            for fig in fig_list: plt.close(fig)
        else:
            plt.show()
    return

def plot_intensity_multi(data_path_list, plot_dict, plot_path, extra_path = "", save_bool=False):
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
    hyst_x_array_list = []

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
            hyst_x_array_list.append(np.load(f))
    
    if len(hyst_x_array_list[0])!=0:
        return

    fig_list = []
    colors = plot_dict["colors"]
    keys = [key for key in colors]
    markers = plot_dict["markers"]
    intensity_dict = plot_dict["intensity_dict"]

    with plt.rc_context({'font.family': intensity_dict["text_font_family"], 'font.size': intensity_dict["text_font_size"], 'mathtext.fontset': intensity_dict["text_font_set"], 'mathtext.rm': intensity_dict["text_font_family"], 'lines.linewidth': intensity_dict["line_width"]}):
        # Plot raw intensity:
        fig0 = plt.figure()
        for i in range(N):
            t = np.arange(len(intensity_array_list[i]))
            plt.step(t, intensity_array_list[i], label = "Run "+str(i), c=colors[keys[i]]) # ASI Raw Intensity
        plt.xlabel('Time [steps]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig0.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig0)

        # Plot normalization intensity:
        fig1 = plt.figure()
        for i in range(N):
            t = np.arange(len(intensity_array_list[i]))
            plt.step(t, normalized_array_list[i], label = "Run "+str(i), c=colors[keys[i]]) # Normalization intensity
        plt.xlabel('Time [steps]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig1.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig1)

        # Plot normalized ASI intensity:    
        fig2 = plt.figure()
        for i in range(N):
            t = np.arange(len(intensity_array_list[i]))
            plt.step(t, norm_intensity_array_list[i], label = "Run "+str(i), c=colors[keys[i]]) # Normalized ASI Intensity
        if len(min_max_list) != 0:
            avg_norm_min_inten = np.mean([i[2] for i in min_max_list])
            avg_norm_max_inten = np.mean([i[5] for i in min_max_list])
            plt.axhline(y = avg_norm_min_inten, label = "Avg Min", color = 'b', linestyle = '--') 
            plt.axhline(y = avg_norm_max_inten, label = "Avg Max", color = 'r', linestyle = '--') 
        plt.xlabel('Time [steps]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig2.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig2)

        # Plot normalized min/max values:
        fig3 = plt.figure()
        min_max_diff = []
        for i in range(N):
            if i==0:
                plt.scatter(i, min_max_list[i][2], label = "Min", color = 'b', marker="^")
                plt.scatter(i, min_max_list[i][5], label = "Max", color = 'r', marker="v")
            else:
                plt.scatter(i, min_max_list[i][2], color = 'b', marker="^")
                plt.scatter(i, min_max_list[i][5], color = 'r', marker="v")
            min_max_diff.append(min_max_list[i][5]-min_max_list[i][2])
        plt.xlabel('Run idx [steps]')
        plt.ylabel('Mean pixel value [unitless]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig3.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig3)

        # Plot shifted raw intensity:
        fig4 = plt.figure()
        for i in range(N):
            if len(shift_intensity_array_list[0])!=0:
                t = np.arange(len(intensity_array_list[i]))
                plt.step(t, shift_intensity_array_list[i], label = "Run "+str(i), c=colors[keys[i]]) # Shifted Raw Intensity
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig4.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig4)

        # Plot shifted normalization intensity
        fig5 = plt.figure()
        for i in range(N):
            if len(shift_normalized_array_list[0])!=0:
                t = np.arange(len(intensity_array_list[i]))
                plt.step(t, shift_normalized_array_list[i], label = "Run "+str(i), c=colors[keys[i]])  # Shifted Normalization intensity
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig5.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig5)

        # Plot shifted normalized ASI intensity
        fig6 = plt.figure()
        zero = []
        for i in range(N):
            if len(shift_norm_intensity_array_list[0])!=0:
                zero_idx = []
                for j in range(len(shift_norm_intensity_array_list[i])-1):
                    if shift_norm_intensity_array_list[i][j]*shift_norm_intensity_array_list[i][j+1]<0:
                        zero_idx.append(j)
                zero.append(zero_idx)
                t = np.arange(len(intensity_array_list[i]))
                plt.step(t, shift_norm_intensity_array_list[i], label = "Run "+str(i), c=colors[keys[i]])  #Shifted Normalized ASI Intensity
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig6.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig6)

        fig7 = plt.figure()
        for i in range(N):
            if len(shift_norm_intensity_array_list[0])!=0:
                t = np.arange(len(intensity_array_list[i]))
                plt.scatter(t, shift_norm_intensity_array_list[i], label = "Run "+str(i), color=colors[keys[i]], s=9, marker=markers[i], alpha=0.6)  #Shifted Normalized ASI Intensity
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig7.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig7)

        # Plot only result with clocked_dynamics_paper results:
        x = np.arange(1,51)
        y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
        fig8 = plt.figure()
        for i in range(N):
            if len(shift_norm_intensity_array_list[0])!=0:
                plt.step(t, shift_norm_intensity_array_list[i], label = "Run "+str(i), c=colors[keys[i]])
        #plt.step(x,y, label= "Reference", c=colors["o"])
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig8.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig8)

        x = np.arange(1,51)
        y = [-0.9785522783816807, -0.9506704202743085, -0.933512095709836, -0.9120643740915165, -0.8970511162285099, -0.8734584915418546, -0.8584450700457181, -0.8369973484273986, -0.8112602297552325, -0.7898125081369132, -0.7833780443816003, -0.7662198834502577, -0.7554960226410979, -0.7404826011449614, -0.7190348795266421, -0.6997318973435287, -0.6718498756030264, -0.6396782931755474, -0.618230571557228, -0.5967828499389085, -0.5710455676336121, -0.5367292457709272, -0.5002681026564714, -0.4745308203511751, -0.44879353804587874, -0.40804291606101095, -0.3672922122595782, -0.3115281687785737, -0.27292232713719433, -0.23217158242747904, -0.20643430012218278, -0.1742627176947037, -0.1442359974272781, -0.2836461879463541, -0.39302949456487446, -0.5067024027786542, -0.5710455676336121, -0.6568364541068898, -0.7490616407023503, -0.830563048305216, -0.860589809480924, -0.9142091135267224, -0.933512095709836, -0.9613942810834681, -0.9721181418926279, -0.9828419208852225, -0.9742627176947037, -0.9978552605647941, -0.9957105211295881, -0.9957105211295881]
        fig9 = plt.figure()
        for i in range(N):
            if len(shift_norm_intensity_array_list[0])!=0:
                plt.scatter(t, shift_norm_intensity_array_list[i], label = "Run "+str(i), color=colors[keys[i]], s=9, marker=markers[i], alpha=0.6)
        #plt.scatter(x,y, label= "Reference", color=colors["o"], s=9, marker="d")
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        fig9.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig9)

        # Plot mean result with error:
        t = np.arange(len(intensity_array_list[i]))
        mean = np.mean(shift_norm_intensity_array_list, axis=0)
        std = np.std(shift_norm_intensity_array_list, axis=0)
        fig10 = plt.figure()
        plt.errorbar(t, mean, std, fmt="o", c="black", ms=2.2, elinewidth=1, label="Mean and std", capsize=1.5, capthick=0.5)
        # plt.scatter(x,y, label= "Reference", color=colors["o"], s=10, marker="d")
        plt.ylim(-1.1,1.1)
        plt.xlabel('Time [steps]')
        plt.ylabel('Magnetization [$M/M\mathrm{s}$]')
        plt.grid(axis=intensity_dict["grid_axis"], color=intensity_dict["grid_color"], linestyle=intensity_dict["grid_linestyle"], linewidth=intensity_dict["grid_linewidth"])
        #fig10.legend(loc=intensity_dict["legend_loc"], bbox_to_anchor=intensity_dict["legend_bbox_to_anchor"]) 
        fig_list.append(fig10)
        
        with open(plot_path + "Data" + extra_path + ".txt", 'w') as f:
            f.write("Min-max diff: [" + "".join(str(i)+", " for i in min_max_diff) + "] \n")
            f.write("Cross zero (idx): " + "".join("[" + str(i[0])+","+str(i[1])+"] " for i in zero) + "\n")
            f.write("Mean: " + str(mean) + "\n")
            f.write("Std: " + str(std) + "\n")
            f.close()

        if save_bool:
            dpi= 200
            fig0.savefig(plot_path + "Raw_intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig1.savefig(plot_path + "Normalization_intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig2.savefig(plot_path + "Normalized_ASI_intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig3.savefig(plot_path + "Normalized_min_max_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig4.savefig(plot_path + "Shifted_Raw_Intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig5.savefig(plot_path + "Shifted_Normalization_Intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig6.savefig(plot_path + "Shifted_Normalized_ASI_Intensity_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig7.savefig(plot_path + "Shifted_Normalized_ASI_Intensity_scatterplot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig8.savefig(plot_path + "Compare_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig9.savefig(plot_path + "Compare_scatterplot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            fig10.savefig(plot_path + "Compare_mean_plot" + extra_path + ".png", dpi=dpi, bbox_inches="tight")
            for fig in fig_list: plt.close(fig)
        else:
            plt.show()
    return

def compare_shift():
    with open("C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Data/23.11.13_samp1 11.8 AB 30.npy", 'rb') as f:
        R30_mask_remove_idx_list = np.load(f)
        R30_norm_xyd = np.load(f)
        R30_bounds = np.load(f)
        R30_min_max = np.load(f)
        R30_intensity_array = np.load(f)
        R30_normalization_array = np.load(f)
        R30_norm_intensity_array = np.load(f)
        R30_shift_intensity_array = np.load(f)
        R30_shift_normalized_array = np.load(f)
        R30_shift_norm_intensity_array = np.load(f)
        R30_hyst_x_array = np.load(f)

    with open("C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Data/23.11.13_samp1 11.8 AB 32.npy", 'rb') as f:
        R32_mask_remove_idx_list = np.load(f)
        R32_norm_xyd = np.load(f)
        R32_bounds = np.load(f)
        R32_min_max = np.load(f)
        R32_intensity_array = np.load(f)
        R32_normalization_array = np.load(f)
        R32_norm_intensity_array = np.load(f)
        R32_shift_intensity_array = np.load(f)
        R32_shift_normalized_array = np.load(f)
        R32_shift_norm_intensity_array = np.load(f)
        R32_hyst_x_array = np.load(f)

    with open("C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Data/23.11.29_samp1 11.8 AB ab 33 Run 0.npy", 'rb') as f:
        R33_mask_remove_idx_list = np.load(f)
        R33_norm_xyd = np.load(f)
        R33_bounds = np.load(f)
        R33_min_max = np.load(f)
        R33_intensity_array = np.load(f)
        R33_normalization_array = np.load(f)
        R33_norm_intensity_array = np.load(f)
        R33_shift_intensity_array = np.load(f)
        R33_shift_normalized_array = np.load(f)
        R33_shift_norm_intensity_array = np.load(f)
        R33_hyst_x_array = np.load(f)
    
    plt.step(np.arange(len(R30_norm_intensity_array)), R30_norm_intensity_array)
    plt.step(np.arange(len(R32_norm_intensity_array)), R32_norm_intensity_array)
    plt.step(np.arange(len(R33_norm_intensity_array)), R33_norm_intensity_array)
    plt.show()

    R30_shift_norm_intensity_array = -(2*R30_norm_intensity_array- R33_min_max[2] - R33_min_max[5])/(R33_min_max[5]-R33_min_max[2])
    R32_shift_norm_intensity_array = -(2*R32_norm_intensity_array- R33_min_max[2] - R33_min_max[5])/(R33_min_max[5]-R33_min_max[2])

    
    plt.step(np.arange(len(R30_shift_norm_intensity_array)), R30_shift_norm_intensity_array)
    plt.step(np.arange(len(R32_shift_norm_intensity_array)), R32_shift_norm_intensity_array)
    plt.step(np.arange(len(R33_shift_norm_intensity_array)), R33_shift_norm_intensity_array)
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
    6: correction_xyd \n
    7: rows \n
    8: cols \n
    9: plot_dict \n
    10: annotate \n
    """
    
    image_folder_path = "C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/"
    data_path = "C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Python/Data/"
    plot_path = "C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Python/Plots/"
    hysteresis_txt = ""
    correction_xyd = [470,80,100]
    palette_tab10 = sns.color_palette("colorblind", 10) 

    plot_dict = {"correction_dict" : {"scalebar_text_x": 25, "scalebar_text_y": 34, "scalebar_color": "black", "scalebar_text": "10 $\mathrm{\mu}$m",
                "scalebar_x": 7, "scalebar_y": 44, "scalebar_w": 86, "scalebar_h": 10, 
                "text_font_size": 12.5, "text_font_family": "Times New Roman", "text_font_set": "custom", "text_shift_x": 0, "text_shift_y": -2},
                
                "images_dict": {"scalebar_text_x": 31, "scalebar_text_y": 42, "scalebar_color": "black", "scalebar_text": "10 $\mathrm{\mu}$m",
                "scalebar_x": 11, "scalebar_y": 52, "scalebar_w": 86, "scalebar_h": 10, 
                "text_font_size": 12.5, "text_font_family": "Times New Roman", "text_font_set": "custom", "text_shift_x": 2, "text_shift_y": 15, "text_color": "w",
                "bbox_facecolor": 'gray', "bbox_edgecolor": 'none', "bbox_pad": 1.0, "bbox_alpha": 0.7}, 
                
                "images_multi_dict": {"text_horizontal_x": 35, "text_horizontal_y": -4, "text_vertical_x": -72, "text_vertical_y": 60, 
                "text_font_size": 12.5, "text_font_family": "Times New Roman", "figsize_pad_x": 0.5}, 
                
                "intensity_dict": {"grid_axis": 'both', "grid_color": "grey", "grid_linestyle": "--", "grid_linewidth": "0.25", 
                "legend_loc": 'center left', "legend_bbox_to_anchor": (0.9, 0.6), "text_font_size": 12.5, "text_font_family": "Times New Roman", "text_font_set": "custom", "line_width": 0.8}, #12.5

                "colors": {"b":palette_tab10[0],"brown":palette_tab10[5], "i":palette_tab10[9], "r":palette_tab10[3], "black": 'black', "g":palette_tab10[2], "purple":palette_tab10[4], "gray":palette_tab10[7], "y":palette_tab10[8], "p":palette_tab10[6], "o":palette_tab10[1]},
                
                "markers": ["s", "D", "P", "X", "*", "o", "v", "^", "<", ">"]
                } 
    
    images_dict_big = {"scalebar_text_x": 62, "scalebar_text_y": 84, "scalebar_color": "black", "scalebar_text": "20 $\mathrm{\mu}$m",
                "scalebar_x": 11, "scalebar_y": 104, "scalebar_w": 192, "scalebar_h": 20, 
                "text_font_size": 12.5, "text_font_family": "Times New Roman", "text_font_set": "custom", "text_shift_x": 4, "text_shift_y": 30, "text_color": "w", 
                "bbox_facecolor": 'gray', "bbox_edgecolor": 'none', "bbox_pad": 1.0, "bbox_alpha": 0.7}

            
    annotate = {}

    if case==-5:
        name = "23.11.13_samp1/11.8 hyst 22deg"
        hysteresis_txt = image_folder_path + name + "/file name.txt"
        mask_remove_idx_list = [-1]
        gif_interval = 300
        rows, cols = 3, 3 
        annotate = [55,56,57,58,120,121,122,123]
        run_path = ""
    elif case==-4:
        name = "23.11.13_samp1/11.8 hyst 0deg"
        hysteresis_txt = image_folder_path + name + "/file name.txt"
        mask_remove_idx_list = [-1]
        gif_interval = 300
        rows, cols = 3, 3 
        annotate = [48,49,50,51,113,114,115,116] 
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
        annotate = {0: "Initial, $t$=", 1: "AB, $t$=", 36: "ab, $t$="} 
    elif case==1:
        name = "23.11.29_samp1/11.8 AB ab 33 Manually"
        mask_remove_idx_list = [0,2,1]
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, $t$=", 1: "AB, $t$=", 36: "ab, $t$="} 
        run_path = ""
    elif case==2:
        name = "23.11.29_samp1/11.8 A B AB A B 33"
        mask_remove_idx_list = [0,2,1]
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, $t$=", 1: "A, $t$=", 11: "B, $t$=", 21: "AB, $t$=", 31: "A, $t$=", 41: "B, $t$="}
    elif case==3:
        name = "23.11.29_samp1/12.8 AB ab 32"
        mask_remove_idx_list = [0,2,1]
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, $t$=", 1: "AB, $t$=", 36: "ab, $t$="}
        plot_dict["images_dict"] = images_dict_big
    elif case==4: 
        name = "23.11.29_samp1/12.8 aAbB 33"
        mask_remove_idx_list = [0] #Forgot normalization min/max!
        gif_interval = 300
        rows, cols = 8,7    
        annotate = {0: "Initial, $t$=", 1: "aAbB, $t$=", 36: "AaBb, $t$="} 
        plot_dict["images_dict"] = images_dict_big
    
    image_folder_path += name + run_path + "/*.png"
    data_path += name.replace("/"," ") + run_path.replace("/"," ") + ".npy"
    plot_path += name.replace("/"," ") + run_path + "/"
    return image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, correction_xyd, rows, cols, plot_dict, annotate



def main():
    # Constants:
    run_path = "/Run "
    multi_plot_path = "C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Python/Plots/"
    extra_path = ""
    
    # Variables:
    gen_data = False
    save_bool = False
    run_mode = 0        #0: Show images single, 1: Plot intensity single, 2: Show images multi, 3: plot multi, 4: histogram
    
    case, run_num = 0, 0 
    # 23.11.13: -5: 11.8 hyst 22deg (-), -4: 11.8 hyst 0deg (-), -3: 11.8 AB 32.5 (-), -2: 11.8 AB 32 (-), -1: 11.8 AB 30 (-)
    # 23.11.29: 0: 11.8 AB ab 33 (0-9), 1: 11.8 AB ab 33 Manually (0), 2: 11.8 A B AB A B 33 (0), 3: 12.8 AB ab 32 (0 & 0 J), 4: 12.8 aAbB 33 (0)
    
    multi_plot_path += "11.8 AB ab 33 Run 0-9/"
    cases_runs_dict = {0:[i for i in range(10)]}

    # multi_plot_path += "11.8 AB ab 30 & 32 & 33/"
    # cases_runs_dict = {-1:[0], -2: [0], 0: [0]}

    # multi_plot_path += "11.8 hyst 0 & 22/"
    # cases_runs_dict = {-4: [0], -5: [0]} 

    hyst_names = ["0 Deg", "22 Deg"]
    names = ["30 mT", "32 mT", "33 mT"]
    image_idx = [5,15,25,35,45]

    

    if run_mode==0:
        # Show images single:
        info = get_info(case,run_path+str(run_num))
        image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, correction_xyd, rows, cols, plot_dict, annotate = info
        if not os.path.exists(plot_path) and save_bool:
            os.makedirs(plot_path)
        raw_image_array, corrected_image_array, cropped_corrected_image_array, bounds, correction_0 = get_images(image_folder_path=image_folder_path, mask_remove_idx_list=mask_remove_idx_list, correction_xyd=correction_xyd)
        
        # make_gifs(image_array=corrected_image_array, gif_interval=gif_interval, plot_path=plot_path, extra_path = extra_path, save_bool=save_bool)
        # make_gifs(image_array=cropped_corrected_image_array, gif_interval=gif_interval, plot_path=plot_path, extra_path = "_cropped" + extra_path, save_bool=save_bool)
        show_correction_box(image_array=raw_image_array, correction_xyd=correction_xyd, rows=rows, cols=cols, correction_dict=plot_dict["correction_dict"], plot_path=plot_path, extra_path=extra_path, save_bool=save_bool)
        # show_images_cropped(image_array=cropped_norm_image_array, rows=rows, cols=cols, images_dict=plot_dict["images_dict"], annotate=annotate, plot_path=plot_path, extra_path = extra_path, save_bool = save_bool)
        # show_hyst_cropped(image_array=cropped_norm_image_array, rows=rows, cols=cols, images_dict=plot_dict["images_dict"], annotate=annotate, plot_path=plot_path, extra_path = extra_path, save_bool = save_bool)
        # show_one_image(image_array=raw_image_array)

    elif run_mode==1:
        # Plot intensity single
        info = get_info(case,run_path+str(run_num))
        image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate = info
        if not os.path.exists(plot_path) and save_bool:
            os.makedirs(plot_path)

        if gen_data:
            generate_data(image_folder_path=image_folder_path, hysteresis_txt=hysteresis_txt, data_path=data_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd)

        plot_intensity_single(data_path=data_path, plot_dict= plot_dict, plot_path=plot_path, extra_path = extra_path, save_bool=save_bool)
        #plot_hysteresis_normalized(data_path=data_path, plot_dict= plot_dict, plot_path=plot_path, extra_path = extra_path, save_bool=save_bool)

    elif run_mode==2:
        # Show images multi:
        if not os.path.exists(multi_plot_path) and save_bool:
            os.makedirs(multi_plot_path)
        image_array_array = []
        for key in cases_runs_dict:
            for element in cases_runs_dict[key]:
                info = get_info(key,run_path+str(element))
                image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate = info
                raw_image_array, norm_image_array, cropped_norm_image_array, bounds, norm_0 = get_images(image_folder_path=image_folder_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd)
                image_array_array.append(cropped_norm_image_array[image_idx])

        # show_image_compare_multi(image_array_array=image_array_array, image_idx=image_idx, names=names, images_multi_dict=plot_dict["images_multi_dict"], plot_path=multi_plot_path, extra_path = extra_path, save_bool = save_bool)
        show_image_dist_multi(image_array_array=np.array(image_array_array), image_idx=image_idx, images_multi_dict=plot_dict["images_multi_dict"], plot_path=multi_plot_path, extra_path = extra_path, save_bool = save_bool)
        
    elif run_mode==3:
        # Plot intensity multiple:
        data_path_list = []
        for key in cases_runs_dict:
            for element in cases_runs_dict[key]:
                info = get_info(key,run_path+str(element))
                image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate = info
                if gen_data:
                    generate_data(image_folder_path=image_folder_path, hysteresis_txt=hysteresis_txt, data_path=data_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd)
                data_path_list.append(data_path)
        
        if not os.path.exists(multi_plot_path) and save_bool:
            os.makedirs(multi_plot_path)
        plot_intensity_multi(data_path_list=data_path_list, plot_dict=plot_dict, plot_path=multi_plot_path, extra_path=extra_path, save_bool=save_bool)
        plot_hysteresis_multi(data_path_list=data_path_list, plot_dict=plot_dict, hyst_names=hyst_names, plot_path=multi_plot_path, extra_path=extra_path, save_bool=save_bool)
    
    elif run_mode==4:
        info = get_info(case,run_path+str(run_num))
        image_folder_path, data_path, plot_path, hysteresis_txt, gif_interval, mask_remove_idx_list, norm_xyd, rows, cols, plot_dict, annotate = info
        # if not os.path.exists(plot_path) and save_bool:
        #     os.makedirs(plot_path)
        raw_image_array, norm_image_array, cropped_norm_image_array, bounds, norm_0 = get_images(image_folder_path=image_folder_path, mask_remove_idx_list=mask_remove_idx_list, norm_xyd=norm_xyd)
        plot_pixel_hist(cropped_norm_image_array, norm_0, plot_dict)
       
    elif run_mode==5:
        compare_shift()
    return

main()




# palette_tab10 = sns.color_palette("colorblind", 10)
# plt.rcParams['axes.prop_cycle'] = cycler(color=palette_tab10)
# for i in range(10):
#     t = np.arange(30)
#     y = np.arange(30)/5
#     plt.plot(t,y+i*3)
# plt.show()

