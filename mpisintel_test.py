import sys
sys.path.append('..')
sys.path.append('../datasets')
import os.path
import glob
from listdataset import ListDataset,ListDataset_test
# from listdataset_test import ListDataset_test



'''
Dataset routines for MPI Sintel.
http://sintel.is.tue.mpg.de/
clean version imgs are without shaders, final version imgs are fully rendered
The dataset is not very big, you might want to only pretrain on it for flownet
'''

def make_dataset(dataset_dir, dataset_type='clean'):
    flow_dir = 'flow'
    assert(os.path.isdir(os.path.join(dataset_dir,flow_dir)))
    img_dir = dataset_type
    assert(os.path.isdir(os.path.join(dataset_dir,img_dir)))

    images = []

    for flow_map in sorted(glob.glob(os.path.join(dataset_dir,flow_dir,'*','*.flo'))):
        flow_map = os.path.relpath(flow_map,os.path.join(dataset_dir,flow_dir))

        scene_dir, filename = os.path.split(flow_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')
        frame_nb = int(frame_nb)
        img1 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
        img2 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
        flow_map = os.path.join(flow_dir,flow_map)
        if not (os.path.isfile(os.path.join(dataset_dir,img1)) or os.path.isfile(os.path.join(dataset_dir,img2))):
            continue
        images.append([[img1,img2],flow_map])

        ###from txt
    # with open(dataset_dir+'/'+dataset_type+'_train.txt', 'r') as f:
    #     _trn_IDs = f.readlines() ##是一个列表，列表中每一个元素都是'05625_img1.ppm###05625_img2.ppm###05625_flow.flo\n'
    #     _trn_IDs = [tuple(ID.rstrip().split("###")) for ID in _trn_IDs]  #还是一个列表，但是列表中的元素变成元组，('05625_img1.ppm', '05625_img2.ppm', '05625_flow.flo')
    #     for ids in _trn_IDs:
    #         imagess_tra.append([[dataset_type+'/'+ids[0],dataset_type+'/'+ids[1]],'flow'+'/'+ids[2]])
    # with open(dataset_dir+'/'+dataset_type+'_val.txt', 'r') as f:
    #     _val_IDs = f.readlines() ##是一个列表，列表中每一个元素都是'05625_img1.ppm###05625_img2.ppm###05625_flow.flo\n'
    #     _val_IDs = [tuple(ID.rstrip().split("###")) for ID in _val_IDs]  #还是一个列表，但是列表中的元素变成元组，('05625_img1.ppm', '05625_img2.ppm', '05625_flow.flo')
    #     for ids in _val_IDs:
    #         imagess_val.append([[dataset_type+'/'+ids[0],dataset_type+'/'+ids[1]],'flow'+'/'+ids[2]])
    


    return images


def make_dataset_kitti(dataset_dir):
    images = []
    o = [line.strip().split(' ') for line in open(dataset_dir, 'r')]
    for qq in o:
        images.append([[qq[0],qq[1]],qq[2]])
    return images
def make_dataset_test(dataset_dir, dataset_type='clean'):
    img_dir = dataset_type

    images_test = []
    labels_test = []



    for flow_map in sorted(glob.glob(os.path.join(dataset_dir,img_dir,'*','*.png'))):
        
        flow_map = os.path.relpath(flow_map,os.path.join(dataset_dir,img_dir))

        scene_dir, filename = os.path.split(flow_map)
        no_ext_filename = os.path.splitext(filename)[0]
        prefix, frame_nb = no_ext_filename.split('_')
        frame_nb = int(frame_nb)
        len_ = len(sorted(glob.glob(os.path.join(dataset_dir,img_dir,scene_dir,'*.png'))))
        if frame_nb == len_:
            continue
        img1 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb))
        img2 = os.path.join(img_dir, scene_dir, '{}_{:04d}.png'.format(prefix, frame_nb + 1))
        flow_map = os.path.join(img_dir, scene_dir, '{}_{:04d}.flo'.format(prefix, frame_nb))


        images_test.append([[img1,img2],None])
        labels_test.append(flow_map)



    return images_test,labels_test

def make_dataset_test_kitti(txt):
    samples = [line.strip().split(' ') for line in open(txt, 'r')]





def mpi_sintel_clean(root, transform=None, target_transform=None,
                     co_transform=None):
    test_list = make_dataset(root, 'clean')
    test_dataset = ListDataset(root, test_list, transform, target_transform, co_transform)
    

    return test_dataset,test_list
def kitti2012(root, transform=None, target_transform=None,
                     co_transform=None):
    test_list = make_dataset_kitti(root)
    test_dataset = ListDataset('/home/wujunjie/kitti', test_list, transform, target_transform, co_transform)
    

    return test_dataset,test_list


def mpi_sintel_final(root, transform=None, target_transform=None,
                     co_transform=None,):
    test_list = make_dataset(root, 'final')
    test_dataset = ListDataset(root, test_list, transform, target_transform, co_transform)
    

    return  test_dataset,test_list


def mpi_sintel_both(root, transform=None, target_transform=None,
                    co_transform=None):
    '''load images from both clean and final folders.
    We cannot shuffle input, because it would very likely cause data snooping
    for the clean and final frames are not that different'''
    #assert(isinstance(split, str)), 'To avoid data snooping, you must provide a static list of train/val when dealing with both clean and final.'
    ' Look at Sintel_train_val.txt for an example'

    test_list1, clean_pre_path = make_dataset_test(root, 'clean')
    test_list2, final_pre_path = make_dataset_test(root, 'final')
    test_dataset = ListDataset_test(root, test_list1 + test_list2, transform, target_transform, co_transform)
    return test_dataset,clean_pre_path + final_pre_path

def kitti(root,txt, transform=None, target_transform=None,
                    co_transform=None):
    make_dataset_test_kitti(txt)
