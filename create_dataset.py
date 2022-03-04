

import os
import torch
from PIL import Image
import numpy as np

import math
import matplotlib.pyplot as plt



def get_boundery_from_dataset(path):

    list_y=[int(i.split("_")[0]) for i in os.listdir(path)]#first part of the name : y
    list_x=[int(i.split("_")[1].split(".")[0]) for i in os.listdir(path)]#second part of the name : x
    xmin,xmax=min(list_x),max(list_x)
    ymin,ymax=min(list_y),max(list_y)

    return {"xmin":xmin,"xmax":xmax,"ymax":ymax,"ymin":ymin}


def pil_loader(path: str) -> Image.Image:
    # from pytorch https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')



class DatasetFromSeparatePatches(torch.utils.data.Dataset):
    
      
    def __init__(self,data_dir,number_of_patch_side=1,transform=None):
        """
        Initialisation of the dataset
        define a new grid where each cell correspound to a 255*number_of_patch_side image with limit xmin,ymin
        the maximum x indice in that grid is given by (xmax-xmin) divided by number_of_patch_size rounded up
        same the maximum y indice
        """
        args_boundery=get_boundery_from_dataset(data_dir)
        
        self.data_dir=data_dir
        xmin=args_boundery["xmin"]
        ymin=args_boundery["ymin"]
        xmax=args_boundery["xmax"]
        ymax=args_boundery["ymax"]
        self.xmin=xmin
        self.ymin=ymin

        self.xmax_new_grid= math.ceil((xmax-xmin)/number_of_patch_side )
        self.ymax_new_grid= math.ceil((ymax-ymin) /number_of_patch_side )

        self.transform = transform
        self.number_of_patch=number_of_patch_side**2
        self.number_of_patch_side=number_of_patch_side

        self.width=256*number_of_patch_side
        self.height=256*number_of_patch_side

    def __len__(self):
      'total number of sample : number of possible sample // number_of_patches rounded up'
      return self.xmax_new_grid*self.ymax_new_grid
      
    
    def __getitem__(self, index):
        
        """
        'Generates one sample of data'
        input : index between 0 and len(dataset)-1 
        first step : convert this index into x,y indices for the new grid
        second step : convert them to get the indice for original grid
        third step : create an empty image
        fourth step : get all the image & put them in the right place
        
        """
        #indice of the return image in the new grid
        indice_y_new_grid=index%self.xmax_new_grid
        indice_x_new_grid=index//self.xmax_new_grid

        
        # indices of the saved image
        
        indices_x_in_saved_image=[self.xmin+indice_x_new_grid*self.number_of_patch_side+i%self.number_of_patch_side for i in range(self.number_of_patch)]
        indices_y_in_saved_image=[self.ymin+indice_y_new_grid*self.number_of_patch_side+i//self.number_of_patch_side for i in range(self.number_of_patch)]

        #indices correspounding in the returned numpy array
        #there is a 
        indices_y_in_numpy_image=[256*(i%self.number_of_patch_side) for i in range(self.number_of_patch)]
        indices_x_in_numpy_image=[256*(i//self.number_of_patch_side) for i in range(self.number_of_patch)]
        
        
        image=np.zeros((self.height,self.width,3),dtype=np.uint8)

        for indice_x_saved_image,indice_y_saved_image,indice_x_numpy_image,indice_y_numpy_image in zip(indices_x_in_saved_image,indices_y_in_saved_image,indices_x_in_numpy_image,indices_y_in_numpy_image):
          # Select sample
        
          path = self.data_dir+f"{indice_y_saved_image}_{indice_x_saved_image}.jpg"
          # Load data and get label
          try:
            image[indice_y_numpy_image:256+indice_y_numpy_image,indice_x_numpy_image:256+indice_x_numpy_image,:]=np.array(pil_loader(path))
 
          except FileNotFoundError:# if the file is not download, put 0
            
            image[indice_y_numpy_image:256+indice_y_numpy_image,indice_x_numpy_image:256+indice_x_numpy_image,:]=np.zeros((256,256,3))
          
        
        if self.transform:
            image = self.transform(image)
        return  image
    def save(self,index,path):
      image=self[index]
      if image.shape[0]==3:#if the image is a tensor
        image=(image.permute(1,2,0).numpy()*255).astype(np.uint8)
      #if not(os.path.isdir(f"{path}/{self.number_of_patch_side}")):
      
      # os.makedirs(f"{path}/{self.number_of_patch_side}")

      Image.fromarray(image).save(f"{path}/{index}_ref.jpg")

    def show(self,index):
      

      image=self[index]
      if image.shape[0]==3:#if the image is a tensor
        plt.imshow(image.permute(1,2,0).numpy())
      else:
        plt.imshow(image)
    