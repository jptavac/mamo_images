
import matplotlib.pyplot as plt

def image(image,size=(10,10),cmap='gray'):   
#Show image in axis.
    figure, axis = plt.subplots(1, 1)               
    figure.set_size_inches(size)                    
    axis.imshow(image,cmap=cmap)

def two_images(img_1, img_2, size=(10,10), title='', cmap ='gray'): 
# Show two images side by side. 
    figure, axis = plt.subplots(1, 2)
    figure.set_size_inches(size)
    figure.suptitle(title)
    axis[0].imshow(img_1, cmap=cmap)
    axis[1].imshow(img_2, cmap=cmap)

def N_images(image_list, size=(10,10), title='', subplot_title_list = '', cmap ='gray'): 
# Show N images side by side.
    N = len(image_list)                             # Obtain image list size.
    figure, axis = plt.subplots(1, N)               # Create subplots according to list size.
    figure.set_size_inches(size)                    
    figure.suptitle(title)                      

    if (subplot_title_list=='') :                   # Show images in list if no subtitles are indicated.
        for index in range(0,N):                        
            axis[index].imshow(image_list[index],cmap=cmap)
    else:                                           # Show images in list with indicated subtitles.
        for index in range (0,N):                   
            axis[index].imshow(image_list[index],cmap=cmap)
            axis[index].title.set_text(str(subplot_title_list[index]))    
