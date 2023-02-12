# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import os
import sys
import io
import glob
import pickle
import argparse
import threading
import queue
import traceback
import numpy as np
import scipy.ndimage
import PIL.Image
import h5py # conda install h5py

from sklearn.model_selection import StratifiedKFold
from skimage.transform import resize
from utils import segment_ChanVese, transform_training

#----------------------------------------------------------------------------

class HDF5Exporter:
    def __init__(self, h5_filename, resolution, channels=3):
        rlog2 = int(np.floor(np.log2(resolution)))
        assert resolution == 2 ** rlog2
        self.resolution = resolution
        self.channels = channels
        self.h5_file = h5py.File(h5_filename, 'w')
        self.h5_lods = []
        self.buffers = []
        self.buffer_sizes = []
        for lod in range(rlog2, -1, -1):
            r = 2 ** lod; c = channels
            bytes_per_item = c * (r ** 2)
            chunk_size = int(np.ceil(128.0 / bytes_per_item))
            buffer_size = int(np.ceil(512.0 * np.exp2(20) / bytes_per_item))
            #change to channel last
            lod = self.h5_file.create_dataset('data%dx%d' % (r,r), shape=(0,r,r,c), dtype=np.uint8,
                maxshape=(None,r,r,c), chunks=(chunk_size,r,r,c), compression='gzip', compression_opts=4)
            
            self.h5_lods.append(lod)
            self.buffers.append(np.zeros((buffer_size,r,r,c), dtype=np.uint8))
            self.buffer_sizes.append(0)

    def close(self):
        for lod in range(len(self.h5_lods)):
            self.flush_lod(lod)
        self.h5_file.close()

    def add_images(self, img):
        assert img.ndim == 4 and img.shape[1] == self.channels and img.shape[2] == img.shape[3]
        assert img.shape[2] >= self.resolution and img.shape[2] == 2 ** int(np.floor(np.log2(img.shape[2])))
        for lod in range(len(self.h5_lods)):
            while img.shape[2] > self.resolution / (2 ** lod):
                img = img.astype(np.float32)
                img = (img[:, :, 0::2, 0::2] + img[:, :, 0::2, 1::2] + img[:, :, 1::2, 0::2] + img[:, :, 1::2, 1::2]) * 0.25
            quant = np.uint8(np.clip(np.round(img), 0, 255))
            ofs = 0
            while ofs < quant.shape[0]:
                num = min(quant.shape[0] - ofs, self.buffers[lod].shape[0] - self.buffer_sizes[lod])
                self.buffers[lod][self.buffer_sizes[lod] : self.buffer_sizes[lod] + num] = quant[ofs : ofs + num]
                self.buffer_sizes[lod] += num
                if self.buffer_sizes[lod] == self.buffers[lod].shape[0]:
                    self.flush_lod(lod)
                ofs += num

    def add_images_channel_last(self, img):
        assert img.ndim == 4 and img.shape[3] == self.channels and img.shape[1] == img.shape[2]
        assert img.shape[2] >= self.resolution and img.shape[2] == 2 ** int(np.floor(np.log2(img.shape[2])))
        for lod in range(len(self.h5_lods)):
            while img.shape[2] > self.resolution / (2 ** lod):
                img = img.astype(np.float32)
                img =   (img[:,  0::2, 0::2, :] + 
                        img[:,  0::2, 1::2,  :] + 
                        img[:,  1::2, 0::2,  :] + 
                        img[:,  1::2, 1::2,  :]) * 0.25

            quant = np.uint8(np.clip(np.round(img), 0, 255))
            ofs = 0
            while ofs < quant.shape[0]:
                num = min(quant.shape[0] - ofs, self.buffers[lod].shape[0] - self.buffer_sizes[lod])
                
                #print("self.buffers.shape:",self.buffers[0].shape)

                self.buffers[lod][ self.buffer_sizes[lod] : self.buffer_sizes[lod] + num] = quant[ofs : ofs + num]
                self.buffer_sizes[lod] += num
                if self.buffer_sizes[lod] == self.buffers[lod].shape[0]:
                    self.flush_lod(lod)
                ofs += num

    def num_images(self):
        return self.h5_lods[0].shape[0] + self.buffer_sizes[0]
        
    def flush_lod(self, lod):
        num = self.buffer_sizes[lod]
        if num > 0:
            self.h5_lods[lod].resize(self.h5_lods[lod].shape[0] + num, axis=0)
            self.h5_lods[lod][-num:] = self.buffers[lod][:num]
            self.buffer_sizes[lod] = 0

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.type, self.value = sys.exc_info()[:2]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func, verbose_exceptions=True): # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            if verbose_exceptions:
                print('\n\nWorker thread caught an exception:\n' + result.traceback + '\n', end=' ')
            raise result.type(result.value)
        return result, args

    def finish(self):
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)
           
        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1
    
        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------

def get_image(path, new_shape):
    try:
        img = np.asarray(PIL.Image.open(path))
        img = resize(img, new_shape, 0)
        img = img * 255
        print("ok", path)
        return img
    except:
        print("error in", path)
        return None

def save_file_h5(h5_path, images, shape):
    h5 = HDF5Exporter(h5_path, shape[0], 3)
    for idx in range(len(images)):
        img = images[idx]

        # assert img.shape == (218, 178, 3)
        # img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
        #img = img.transpose(2, 0, 1) # HWC => CHW
        h5.add_images_channel_last(img[np.newaxis])

    print('%-40s\r' % 'Flushing data...', end=' ')
    h5.close()

def create_custom_channel_last(root, output_root, new_shape=(512,512)):

    datasets = os.listdir(root)
    datasets.sort()

    for data_name in datasets:

        print("start", data_name)

        x = [os.path.join(r,f) for r,d,files in os.walk( os.path.join(root,data_name) ) for f in files if "-nocluster" not in f]
        x.sort()
        
        # load images only once
        x = [ (get_image(i, new_shape), i) for i in x ]
        # labels, benign = 0, 1 otherwise
        y = np.asarray([ 0 if "benign" in p.split("/")[-2] else 1 for i, p in x if i is not None])
        x = np.asarray( [ i for i, p in x if i is not None] )

        # kfold
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        c_fold = -1
        for train, test in kfold.split(x, y):
            c_fold += 1

            # stop earlier
            if c_fold == 5:
                break

            # train
            c_images = x[train]
            c_y = y[train]

            g_name = os.path.join(output_root, "-".join([data_name, str(new_shape[0]), str(c_fold)]) )

            # select less represented class to augment in next phase
            unique, counts = np.unique(c_y, return_counts=True)
            less_label = unique[np.argmin(counts)]

            c_images_train = c_images[ np.argwhere(c_y == less_label).flatten() ]
            
            save_file_h5( "-".join([g_name, "train.h5"]), c_images_train, new_shape)
            
            # all images model training
            c_images = transform_training(c_images)
            
            np.save( "-".join([g_name, "train", "X.npy"]), c_images)
            np.save( "-".join([g_name, "train", "Y.npy"]), c_y)

            # test
            c_images = x[test]
            c_images = transform_training(c_images)
            c_y = y[test]
            
            np.save( "-".join([g_name, "test", "X.npy"]), c_images)
            np.save( "-".join([g_name, "test", "Y.npy"]), c_y)

#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating, extracting, and visualizing HDF5 datasets.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)
        
    subparsers = parser.add_subparsers(dest='command')
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p.add_argument(     'h5_filename',      help='HDF5 file to inspect')

    p = add_command(    'create_celeba_channel_last',    'Create HDF5 channel last dataset for CelebA.',
                                            'create_celeba celeba-128x128.h5 ~/celeba')

    p.add_argument(     'h5_filename',      help='HDF5 file to create')
    p.add_argument(     'celeba_dir',       help='Directory to read CelebA data from')
    p.add_argument(     '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(     '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    args = parser.parse_args(argv[1:])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

# if __name__ == "__main__":
#     execute_cmdline(sys.argv)


#create_celeba_channel_last('datasets/celeba_128x128.h5', 'datasets/img_align_celeba', cx=89, cy=121)

create_custom_channel_last(
    '/datasets/01-classified-test',
    '/datasets',
    new_shape=(512,512) 
)

#----------------------------------------------------------------------------
