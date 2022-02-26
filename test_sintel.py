import torch
import sys
sys.path.append('..')
import torchvision.transforms as transforms
from mpisintel_test import mpi_sintel_both,mpi_sintel_clean,mpi_sintel_final,kitti
import torch.nn.functional as F
from util import adapt_xy, AverageMeter,realEPE,ArrayToTensor,Compose
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# from optflow import save_flow
from optflow import flow_to_img,save_flow
from skimage.io import imsave
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np 
from pwc_part.run import Network
import math
from utils.flowlib import flow_to_image
# from . import flowtransforms as transforms
value_scale = 255
mean = [0.485, 0.456, 0.406]
pad_mean = [item * value_scale for item in mean]
std = [0.229, 0.224, 0.225]


CLEAN_OR_FINAL_OR_TEST = 'test'  ##clean,final,test
print (f'It is {CLEAN_OR_FINAL_OR_TEST}')
if CLEAN_OR_FINAL_OR_TEST == 'clean' or CLEAN_OR_FINAL_OR_TEST == 'final':
    Sintel_path = '/media/study/paper/OpticalFlow/DataSet/Sintel/training'
else:
    Sintel_path = '/media/gus/22969CF6969CCC23/MPI-Sintel-testing/test'

MODEL_PATH = './sintel.pth.tar'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_transform = transforms.Compose([
    ArrayToTensor(),
    transforms.Normalize(mean=[0,0,0], std=[255,255,255]),
    transforms.Normalize(mean=mean, std=std)
])

target_transform = transforms.Compose([
    ArrayToTensor(),
    transforms.Normalize(mean=[0,0],std=[20,20])
])

co_transform = None
if CLEAN_OR_FINAL_OR_TEST == 'clean':
    mpi_dataset,pre_list = mpi_sintel_clean(root=Sintel_path, transform=input_transform, 
                                target_transform=target_transform, co_transform=co_transform,
                                )
                         
elif CLEAN_OR_FINAL_OR_TEST == 'final':
    mpi_dataset,pre_list = mpi_sintel_final(root=Sintel_path, transform=input_transform, 
                                target_transform=target_transform, co_transform=co_transform,
                                )

elif CLEAN_OR_FINAL_OR_TEST == 'test':
    mpi_dataset,pre_list = mpi_sintel_both(root=Sintel_path, transform=input_transform, 
                                target_transform=None, co_transform=co_transform,
                                )

    
                         
                            
a = mpi_dataset[0]
print('{} samples found'.format(len(mpi_dataset)))

train_loader = torch.utils.data.DataLoader(
                             mpi_dataset, batch_size=1,
                            num_workers=0, pin_memory=True, shuffle=None)


model= Network()
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint['state_dict'])



model = model.cuda()

model = torch.nn.DataParallel(model)
cudnn.benchmark = True

train_iter = iter(train_loader)

flow2_EPEs = AverageMeter()
model.eval()
steps = len(mpi_dataset)
for i in range(steps):
    if CLEAN_OR_FINAL_OR_TEST == 'clean' or CLEAN_OR_FINAL_OR_TEST == 'final':
        input_images, target = next(train_iter)

        target = target.to(device)
        im1 = input_images[0].to(device)
        im2 = input_images[1].to(device)
        with torch.no_grad():
            pre_path_occ = pre_list[i][1].replace('.flo','.png')

            intWidth = im1.size(3)
            intHeight = im1.size(2)
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

            im1 = torch.nn.functional.interpolate(input=im1, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            im2 = torch.nn.functional.interpolate(input=im2, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            upsampled_output = torch.nn.functional.interpolate(input=model(im1,im2)[0], size=(intHeight, intWidth), mode='bilinear', align_corners=False)

            upsampled_output[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            upsampled_output[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)



            flow2_EPE = 20*realEPE(upsampled_output, target)
            
            flow_png = flow_to_img(upsampled_output.squeeze_().to('cpu').detach().numpy().transpose(1,2,0)*20)
            upsampled_output = upsampled_output.squeeze_().to('cpu').detach().numpy().transpose(1,2,0)*20
            import cv2
            pre_path = pre_list[i][1].replace('.flo','_'+f'{flow2_EPE:.2f}'+'.png')
            

    
        flow2_EPEs.update(flow2_EPE.item(), target.size(0))
        print (f'Step: [{i}/[{steps}]]\t EPE {flow2_EPEs}')

        pre_path = pre_list[i][1].replace('.flo','_'+f'{flow2_EPE:.2f}'+'.png')



    else:
        input_images = next(train_iter)

        im1 = input_images[0].to(device)
        im2 = input_images[1].to(device)


        with torch.no_grad():


            intWidth = im1.size(3)
            intHeight = im1.size(2)
            intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
            intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

            im1 = torch.nn.functional.interpolate(input=im1, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)
            im2 = torch.nn.functional.interpolate(input=im2, size=(intPreprocessedHeight, intPreprocessedWidth), mode='bilinear', align_corners=False)

            upsampled_output = torch.nn.functional.interpolate(input=model(im1,im2)[0], size=(intHeight, intWidth), mode='bilinear', align_corners=False)

            upsampled_output[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
            upsampled_output[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

            upsampled_output = upsampled_output.squeeze_().to('cpu').detach().numpy().transpose(1,2,0)*20
            flow_png = flow_to_img(upsampled_output)


        print (f'Step: [{i}/[{steps}]]\t ')
        pre_path_png = pre_list[i].replace('.flo','.png')
        pre_path_flo = pre_list[i]
        try:
            imsave('./'+'test_png'+'/'+pre_path_png,flow_png)
            save_flow('./'+'test_flo'+'/'+pre_path_flo,upsampled_output)
        except:
            a,_=pre_path_png.split("/frame")
            os.makedirs('./'+'test_png/'+a)
            imsave('./'+'test_png'+'/'+pre_path_png,flow_png)
            os.makedirs('./'+'test_flo/'+a)
            save_flow('./'+'test_flo'+'/'+pre_path_flo,upsampled_output)
    









    
