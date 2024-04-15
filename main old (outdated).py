
import glob
from matplotlib import image as img
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import numpy as np
import scipy.ndimage.measurements as ndi

def outdated():
    def plot_arrows():
        "Gjorde det i powerpoint i stedet."
        num = 5
        a = np.linspace(0,num-1,num)
        x,y = np.meshgrid(a,a)
        x[::2, :] += 1/2
        x = x.flatten()
        y = y.flatten()
        x = np.delete(x, [5,15])
        y = np.delete(y, [5,15])

        u = np.random.uniform(low=-1, high=1, size=num*num-2) 
        v = np.random.uniform(low=-1, high=1, size=num*num-2)
        r = np.power(np.add(np.power(u,2), np.power(v,2)),0.5)

        #plt.scatter(x,y)
        plt.quiver(x,y,u/r,v/r, pivot="mid")
        plt.show()
        return

def make_gifs():
    def update(i):
        im.set_array(image_array[i])
        return im, 

    # Paths in and out:
    fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.09.08_1.sample_big_box/loop-0deg/*.png"
    fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/big_box-loop-0deg.gif"

    # fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.09.08_1.sample_12.8/loop-0deg/*.png"
    # fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/12.8-loop-0deg.gif"

    # fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.09.08_1.sample_12.8/loop-22deg/*.png"
    # fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/12.8-loop-22deg.gif"

    # fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.09.08_1.sample_12.8/AB32/*.png"
    # fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/12.8-AB32.gif"

    # fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.09.29_1.sample_10.3/loop-22deg/*.png"
    # fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/10.3-loop-22deg.gif"
    
    fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 hyst 0deg/*.png"
    fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/23.11.13 samp1 11.8 hyst 0deg.gif"
    
    image_path_list = glob.glob(fp_in)
    mask_path = image_path_list.pop()
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    image_path_list = image_path_list[42:60] + image_path_list[108:122]
    image_array = []

    for image_path in image_path_list:
        image = img.imread(image_path)
        image_array.append(image)
        # image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]])
    shape = image_array[0].shape
    
    # print(len(image_array))
    # print(shape)
   
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/50.0,shape[0]/50.0)
    ax.set_axis_off() # You don't actually need this line as the saved figure will not include the labels, ticks, etc, but I like to include it
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    im = ax.imshow(image_array[0], cmap='Greys_r', animated=True)   #interval=600/300
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=500, blit=True, repeat_delay=10,)

    #plt.show()
    
    animation_fig.save(fp_out) #, savefig_kwargs={'bbox_inches':'tight', 'pad_inches':0}
    
    
    # width, height = im.size
    # left = 5
    # top = height / 4
    # right = 164
    # bottom = 3 * height / 4
    # im1 = im.crop((left, top, right, bottom))
    # im1.show()

    return

def make_gifs_cropped():
    def update(i):
        im.set_array(image_array[i])
        return im, 

    # Paths in and out:
    fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 ab 33/*.png"
    fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Gifs/23.11.13 samp1 11.8 ab 33.gif"
    
    image_path_list = glob.glob(fp_in)
    mask_path = image_path_list.pop(0)
    mask = img.imread(mask_path)
    mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])
    image_array = []

    for image_path in image_path_list:
        image = img.imread(image_path)
        # image_array.append(image)
        image_array.append(image[y_bounds[0]:y_bounds[1],x_bounds[0]:x_bounds[1]])
    shape = image_array[0].shape
    
    # print(len(image_array))
    # print(shape)
   
    fig, ax = plt.subplots()
    fig.set_size_inches(shape[1]/15.0,shape[0]/15.0)
    ax.set_axis_off() # You don't actually need this line as the saved figure will not include the labels, ticks, etc, but I like to include it
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    im = ax.imshow(image_array[0], cmap='Greys_r', animated=True)   #interval=600/300
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=500, blit=True, repeat_delay=10,)

    #plt.show()
    
    animation_fig.save(fp_out) #, savefig_kwargs={'bbox_inches':'tight', 'pad_inches':0}

    return

def plot_hysteresis():
    #file1 = open('C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/08.09.23_1.sample_big_box/loop-0deg/IMGs.txt', 'r')
    file1 = open('C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/29.09.23_1.sample_10.3/loop-22deg/IMGs.txt', 'r')
    Lines = file1.readlines()
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
    #Make nice images!
 
def plot_intensity_and_images(case=0, loop_mask=False):
    extra_path = "1"
    save = True
    mask_idx = 0
    normalization_point = (80,470)
    normalization_box = 100
    AB_num, ab_num = 35,15
    AB_bool = True
    text_fontsize = 8.5
    text_shift = (2,15)

    if case==0:
        fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.09.29_1.sample_10.3/loop-22deg/*.png"
        fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/loop-22deg"
        mask_idx = -1
        rows, cols = 10,7
        loop_mask = True
        AB_bool = False
        normalization_point = (50,470)
    elif case==1:
        fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 AB 30/*.png"
        fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/11.8 AB 30"
        rows, cols = 8,7
        
    elif case==2:
        fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 AB 32/*.png"
        fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/11.8 AB 32"
        rows, cols = 8,7
    elif case==3:
        fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 AB 33/*.png"
        fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/11.8 AB 33"
        rows, cols = 8,7
    elif case==4:
        fp_in = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Measurements/23.11.13_samp1/11.8 AB 32.5/*.png"
        fp_out = "C:/Users/eivhe/OneDrive - NTNU/NTNU/Prosjektoppgave/Python/Plots/11.8 AB 32.5"
        rows, cols = 13,8
        AB_num, ab_num = 70,30
    else:
        return

    # Get file paths and import mask:
    file_paths = glob.glob(fp_in)
    mask_path = file_paths.pop(mask_idx)
    mask = img.imread(mask_path)
    if loop_mask: mask[1,0]= 0.0
    bound = np.argwhere(mask)
    x_bounds = min(bound[:, 1]), max(bound[:, 1])
    y_bounds = min(bound[:, 0]), max(bound[:, 0])

    # Import images and plot intensity:
    image_array = []
    intensity_array = []
    normalized_array = []
    for file in file_paths:
        image = img.imread(file)
        image_array.append(image)
        intensity_array.append(ndi.mean(image, mask))
        normalized_array.append(ndi.mean(image[normalization_point[0]:normalization_point[0]+normalization_box, normalization_point[1]:normalization_point[1]+normalization_box]))
    image_array = np.array(image_array) 
    intensity_array = np.array(intensity_array)
    normalized_array = np.array(normalized_array)
    
    # print(image_array[0].shape)
    # print(x_bounds, y_bounds)

    # Normalization of intensity:
    norm_intensity_array = intensity_array.copy() + normalized_array[0] - normalized_array  #np.amax(normalized_array)
    min_intensity = ndi.mean(image_array[34][251:281, 523:573]) + normalized_array[0] - normalized_array[34]
    max_intensity = intensity_array[0]
    mid_intensity = (min_intensity+max_intensity)/2
    print("min, mid, max intensity: ", min_intensity, mid_intensity, max_intensity)

    # Plot raw intensity:
    fig0 = plt.figure()
    plt.step(np.arange(len(intensity_array)), intensity_array, label = "ASI Intensity")
    # plt.step(np.arange(len(normalized_array)), normalized_array, label = "Normalization intensity")
    # plt.step(np.arange(len(norm_intensity_array)), norm_intensity_array, label = "Normalized ASI Intensity")
    # plt.scatter(34, min_intensity)
    # plt.plot([0,len(image_array)], [mid_intensity,mid_intensity], 'b--')
    fig0.legend()

    # Shift axis to get from -1 to 1:
    normalized_array = (normalized_array - mid_intensity)/((max_intensity-min_intensity)/2)
    intensity_array = (intensity_array - mid_intensity)/((max_intensity-min_intensity)/2)
    norm_intensity_array = (norm_intensity_array - mid_intensity)/((max_intensity-min_intensity)/2)
    normalized_array *= -1
    intensity_array *= -1
    norm_intensity_array *= -1
   

    zero_idx = []
    for j in range(len(norm_intensity_array)-1):
        if norm_intensity_array[j]*norm_intensity_array[j+1]<0:
            zero_idx.append(j)
    print("zero magnetization idx: ", zero_idx)

    # Plot shifted intensity:
    fig1 = plt.figure()
    plt.step(np.arange(len(intensity_array)), intensity_array, label = "ASI Intensity")
    plt.step(np.arange(len(normalized_array)), normalized_array, label = "Normalization intensity")
    plt.step(np.arange(len(norm_intensity_array)), norm_intensity_array, label = "Normalized ASI Intensity")
    plt.ylim(-1.05,1.05)
    fig1.legend()

    # Plot norm box:
    fig2, axs2 = plt.subplots(rows,cols,constrained_layout=True)
    fig2.set_size_inches(cols,rows)
    axs2 = axs2.flatten()
    for i in range(rows*cols):
        if i < len(image_array):
            axs2[i].imshow(image_array[i], cmap='Greys_r') 
            axs2[i].text(normalization_point[1],normalization_point[0],"t="+str(i), fontsize = text_fontsize)
        axs2[i].set_xlim(normalization_point[1], normalization_point[1]+normalization_box) 
        axs2[i].set_ylim(normalization_point[0]+normalization_box, normalization_point[0])
        axs2[i].set_axis_off()

    # Plot images:
    fig3, axs3 = plt.subplots(rows,cols,constrained_layout=True)
    fig3.set_size_inches(cols,rows)
    axs3 = axs3.flatten()
    for i in range(rows*cols):
        if i < len(image_array):
            axs3[i].imshow(image_array[i], cmap='Greys_r') 
        if AB_bool:
            if i==0:
                axs3[i].text(x_bounds[0]+text_shift[0],y_bounds[0]+text_shift[1],"Init state, t="+str(i), fontsize = text_fontsize, color="w")
            elif i<AB_num:
                axs3[i].text(x_bounds[0]+text_shift[0],y_bounds[0]+text_shift[1],"t="+str(i)+", AB", fontsize = text_fontsize, color="w")
            elif i<AB_num+ab_num:
                axs3[i].text(x_bounds[0]+text_shift[0],y_bounds[0]+text_shift[1],"t="+str(i)+", ab", fontsize = text_fontsize, color="w")
        axs3[i].set_xlim(x_bounds)
        axs3[i].set_ylim(y_bounds[1],y_bounds[0])
        axs3[i].set_axis_off()

    if save:
        fig0.savefig(fp_out+" raw_intensity_plot" + extra_path + ".png")
        # fig1.savefig(fp_out+" intensity_plot" + extra_path + ".png")
        # fig2.savefig(fp_out+" norm_box_images" + extra_path + ".png")
        # fig3.savefig(fp_out+" images" + extra_path + ".png")
    else:
        plt.show()
    
    #https://campus.datacamp.com/courses/biomedical-image-analysis-in-python/measurement?ex=5
    return
    

# make_gifs()
# make_gifs_cropped()
# plot_hysteresis()
# plot_intensity_and_images(3)


