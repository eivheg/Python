import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import glob
from matplotlib import image as img
from matplotlib.patches import Rectangle
 
def create_mask():
    # list_29 = ['6.5', '6.6', '6.7', '7.3', '7.5', '7.6', '7.8']
    # image_folder_path = 'C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/24.04.29_IB2403.1/'
    # list_ = list_29

    # list_4 = ['5.2', '5.3', '6.1', '6.2', '6.3', '6.4', '6.5', '6.6', '6.7', '6.8', '6.9', '7.0', '7.1', '7.2', '7.3', '7.4', '7.5', '7.6', '7.7', '7.8', '7.9']
    # image_folder_path = 'C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/24.04.04_IB2403.1/'
    # list_ = list_4

    list_29 = ['6.5']
    image_folder_path = 'C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/24.04.29_IB2403.1/'
    list_ = list_29

    for name in list_:
        image_path_list = glob.glob(image_folder_path + name + '/Run 0/*.png')
        image = img.imread(image_path_list[1])
        
        fig, ax = plt.subplots()
        shape = image.shape
        inches = 10
        fig.set_size_inches(inches*shape[1]/np.min(shape),inches*shape[0]/np.min(shape))
        ax.set_axis_off()
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
        ax.imshow(image, cmap='Greys_r', animated=True) 
        fig.canvas.manager.window.wm_geometry("+%d+%d" % (2000, 0)) #Open on second screen
        yroi = plt.ginput(0,0)
        plt.close()
        # print(yroi)
        print(name)

        # For mask:
        for j in range(0,len(yroi),2):
            x1, y1, x2, y2 = yroi[j][0], yroi[j][1], yroi[j+1][0], yroi[j+1][1]
            print([x1,x2,y1,y2])
            ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, linewidth=0.25, edgecolor='r', facecolor='none'))
        
        # Correction box:
        # for j in range(len(yroi)):
        #     x1, y1, x2, y2 = yroi[j][0], yroi[j][1], yroi[j][0]+100, yroi[j][1]+100
        #     print([x1,y1])
        #     ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, linewidth=0.25, edgecolor='r', facecolor='none'))
        plt.show()

#create_mask()

mask_29 = {6.5 : [[244, 352, 176, 284],[440, 547, 176, 284],[244, 352, 371, 479],[440, 547, 371, 479]], 
        6.6 : [[244, 354, 170, 278],[440, 550, 170, 278],[244, 354, 364, 473],[440, 550, 364, 473]],
        6.7 : [[249, 356, 171, 281],[445, 552, 171, 281],[249, 356, 366, 475],[445, 552, 366, 475]],
        7.3 : [[244, 354, 171, 280],[442, 550, 172, 280],[244, 354, 366, 476],[442, 550, 366, 476]],
        7.5 : [[240, 348, 173, 281],[435, 543, 173, 281],[240, 348, 367, 476],[435, 543, 367, 476]],
        7.6 : [[244, 353, 174, 282],[440, 547, 174, 282],[244, 353, 370, 477],[440, 547, 370, 477]],
        7.8 : [[245, 355, 175, 283],[441, 551, 175, 283],[245, 355, 369, 478],[441, 551, 369, 478]] }

# Må oppdatere for nyeste dataen når vi vet hvilke data som skal vises!
# 6.3 fix, 6.4 3-4, 6.5 5-11, 6.6 2-4, 6.7 3-4, 7.1 3, 7.2 3-6-9, 7.3 0-5 fix, 7.3 6-8, 7.4 5, 7.5 4-6, 7.8 4-5
mask_4 = {5.2 : [[253.837, 360.409, 180.544, 283.264],[449.647, 555.577, 179.902, 284.548],[253.195, 359.125, 371.86, 481.0],[449.005, 555.577, 373.144, 481.642]],
        5.3 : [[246.775, 353.347, 173.482, 277.486],[441.301, 547.231, 173.482, 276.202],[244.849, 351.421, 368.65, 470.728],[440.659, 548.515, 368.65, 473.938]],
        6.1 : [[232.009, 339.223, 168.988, 272.992],[427.177, 535.675, 169.63, 273.634],[232.009, 338.581, 364.156, 468.802],[427.177, 534.391, 365.44, 469.444]],
        6.2 : [[233.935, 341.791, 170.914, 274.276],[429.103, 536.959, 171.556, 275.56],[232.651, 341.149, 364.156, 472.654],[426.535, 536.317, 366.082, 472.654]],
        6.3 : [[235.861, 345.001, 165.778, 274.276],[432.313, 540.169, 167.062, 273.634],[236.503, 344.359, 360.946, 468.802],[431.671, 539.527, 361.588, 469.444]],
        6.4 : [[255.121, 364.261, 153.580, 259.51],[451.573, 558.787, 154.864, 260.794],[255.763, 363.619, 348.748, 454.678],[450.931, 558.787, 348.106, 454.036]],
        6.5 : [[220.453, 329.593, 184.396, 285.832],[415.621, 523.477, 184.396, 286.474],[220.453, 328.309, 378.28, 480.358],[414.979, 523.477, 378.28, 482.284]],
        6.6 : [[216.601, 324.457, 181.828, 289.042],[411.127, 519.625, 181.828, 291.61],[214.675, 323.815, 376.996, 484.852],[411.127, 518.341, 376.996, 486.136]],
        6.7 : [[208.255, 314.827, 177.334, 285.19],[404.065, 511.279, 178.618, 287.116],[206.971, 315.469, 371.86, 481.642],[402.781, 509.995, 373.786, 482.284]],
        6.8 : [[228.799, 337.297, 183.112, 286.474],[423.967, 533.107, 182.470, 287.758],[228.157, 337.297, 377.638, 481.0],[422.683, 532.465, 378.28, 482.926]],
        6.9 : [[266.035, 377.101, 195.952, 301.24],[462.487, 571.627, 196.594, 302.524],[267.319, 375.175, 391.12, 495.766],[461.845, 570.985, 392.404, 497.050]],
        7.0 : [[242.2813, 352.063, 174.124, 283.264],[437.449, 547.873, 175.408, 281.338],[241.639, 351.421, 368.65, 477.148],[438.091, 546.589, 369.292, 478.432]],
        7.1 : [[249.985, 357.841, 177.334, 285.19],[445.153, 552.367, 179.26, 285.19],[248.059, 356.557, 373.786, 480.358],[443.869, 552.367, 374.428, 479.716]],
        7.2 : [[243.565, 350.779, 172.840, 278.77],[438.091, 544.663, 174.766, 279.412],[242.281, 349.495, 370.576, 475.222],[436.807, 544.663, 369.934, 477.148]],
        7.3 : [[210.823, 321.889, 171.556, 277.486],[407.275, 518.341, 173.482, 278.128],[212.107, 316.753, 366.082, 472.012],[405.349, 513.847, 364.798, 472.654]],
        7.4 : [[224.947, 331.519, 197.236, 303.166],[420.757, 527.971, 197.878, 302.524],[223.021, 332.161, 393.046, 497.692],[418.831, 527.329, 393.046, 500.26]],
        7.5 : [[275.665, 382.879, 174.766, 280.054],[468.265, 579.331, 175.408, 280.054],[273.739, 383.521, 369.292, 473.938],[470.191, 578.689, 371.218, 477.790]],
        7.6 : [[226.873, 335.371, 167.704, 276.844],[422.683, 530.539, 169.63, 278.128],[226.231, 334.087, 364.156, 473.296],[422.041, 528.613, 364.156, 473.938]],
        7.7 : [[240.355, 348.853, 174.124, 276.202],[435.523, 544.021, 172.198, 278.128],[239.713, 347.569, 368.008, 472.654],[434.881, 542.737, 368.008, 472.012]],
        7.8 : [[269.887, 376.459, 185.68, 290.326],[465.055, 573.553, 183.112, 289.042],[267.319, 375.817, 379.564, 482.926],[463.129, 570.985, 380.206, 486.136]],
        7.9 : [[292.357, 400.213, 158.074, 265.93],[486.883, 596.665, 158.716, 265.288],[290.431, 399.571, 351.958, 461.0980],[486.883, 594.097, 352.600, 461.098]]}

# mask_ = {6.5: [340,440,40,140], 6.6: [340,440,40,140], 6.7: [340,440,40,140], 7.3: [340,440,40,140], 7.5: [340,440,40,140], 7.6: [340,440,40,140], 7.8: [340,440,40,140]}

mask_ = {7.7: [[240, 348, 172, 280],[435, 543, 172, 280],[240, 348, 366, 474],[435, 543, 366, 474]], 
          7.3 : [[208, 318, 169, 279],[403, 513, 169, 279],[208, 318, 364, 474],[403, 513, 364, 474]],
          6.7 : [[256, 363, 179, 283],[447, 555, 179, 283],[256, 363, 376, 479],[447, 555, 376, 479]]
        }

def test_mask(mask_):
    # list_29 = [6.5, 6.6, 6.7, 7.3, 7.5, 7.6, 7.8]
    # image_folder_path = 'C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/24.04.29_IB2403.1/'
    # list_ = list_29
    # list_run_ = list_run_29

    image_folder_path = 'C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/24.04.04_IB2403.1/'
    list_ = [6.7] #7.3, 7.7
    list_run_ = [202]
    
    # list_4 = [5.2, 5.3, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9]
    # list_run_4 = [4,8,2,2,4,5,12,5,5,2,3,1,4,10,9,6,7,8,6,6,4]
    # image_folder_path = 'C:/Users/eivhe/OneDrive/NTNU/Prosjekt- og masteroppgave/Measurements/24.04.04_IB2403.1/'
    # list_ = list_4
    # list_run_ = list_run_4

    for id,name in enumerate(list_):
        #for run in range(list_run_[id]):
        for run in list_run_:
            #image_path_list = glob.glob(image_folder_path + f'{name}/Run {run}/*.png')
            image_path_list = glob.glob(image_folder_path + f'{name}/Hyst {run}deg/*.png')
            image = img.imread(image_path_list[30])
        
            fig, ax = plt.subplots()
            shape = image.shape
            inches = 10
            fig.set_size_inches(inches*shape[1]/np.min(shape),inches*shape[0]/np.min(shape))
            ax.set_axis_off()
            fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
            ax.imshow(image, cmap='Greys_r', animated=True) 
            #fig.canvas.manager.window.wm_geometry("+%d+%d" % (2000, 0)) #Open on second screen
            
            mask_idx = mask_[name]
            for j in range(4):
                x1, x2, y1, y2 = mask_idx[j][0], mask_idx[j][1], mask_idx[j][2], mask_idx[j][3]
                ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, linewidth=0.25, edgecolor='r', facecolor='none'))
            
            # x1, x2, y1, y2 = mask_idx[0], mask_idx[1], mask_idx[2], mask_idx[3]
            # ax.add_patch(Rectangle((x1,y1), x2-x1, y2-y1, linewidth=0.25, edgecolor='r', facecolor='none'))
            # print(f'{name} {run}')
            plt.show()

test_mask(mask_)










# Full copy:
# mask_29 = {6.5 : [[244.207, 352.705, 176.05, 283.906],[440.017, 547.231, 176.05, 284.548],[242.923, 351.421, 370.576, 479.074],[439.375, 547.231, 371.218, 480.358]], 
#         6.6 : [[244.207, 353.347, 168.988, 276.202],[440.017, 549.799, 169.63, 278.77],[243.565, 353.989, 364.156, 473.296],[440.017, 548.515, 364.156, 473.296]],
#         6.7 : [[248.701, 357.199, 170.272, 280.054],[444.511, 552.367, 170.914, 281.338],[248.701, 355.915, 364.798, 474.58],[444.511, 552.367, 367.366, 474.58]],
#         7.3 : [[245.491, 354.631, 170.272, 280.054],[441.943, 549.799, 171.556, 280.696],[242.923, 353.989, 365.44, 476.506],[440.659, 548.515, 366.724, 475.864]],
#         7.5 : [[240.355, 348.211, 172.840, 280.696],[435.523, 543.379, 173.482, 280.696],[239.071, 346.927, 366.724, 475.864],[434.881, 542.737, 367.366, 476.506]],
#         7.6 : [[245.491, 352.705, 172.840, 281.338],[440.659, 547.231, 174.124, 282.622],[244.207, 352.063, 368.65, 477.790],[439.375, 547.231, 370.576, 477.148]],
#         7.8 : [[244.207, 355.273, 175.408, 283.264],[441.301, 550.441, 174.124, 282.622],[246.775, 355.273, 369.292, 477.790],[441.301, 551.725, 369.292, 477.148]] }
