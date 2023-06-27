import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import os 
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fill50k/prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread('./training/fill50k/' + source_filename)
        target = cv2.imread('./training/fill50k/' + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

def make_dataset(dir):
    #import pdb; pdb.set_trace()
    if os.path.isfile(dir):
        images = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)

    return images

def pil_loader(path):
    return Image.open(path).convert('RGB')

# for flow condition
class FlowbpsDataset(Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[512, 128], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index])
        file_name1 = str(self.flist[index]).replace('.jpg','.png')

        #img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'images', file_name)))
        #cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))
        #cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'images_out', file_name1)))
        
        source = cv2.imread(os.path.join(self.data_root, 'images', file_name))
        target = cv2.imread(os.path.join(self.data_root, 'images_out', file_name1))
        source = cv2.resize(source,(self.image_size[0], self.image_size[1]))
        target = cv2.resize(target,(self.image_size[0], self.image_size[1]))
        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0


        # ret['gt_image'] = img
        # ret['cond_image'] = cond_image
        # ret['path'] = file_name
        prompt = 'a high resolution streetview image'
        return dict(jpg=target, txt= prompt, hint=source)

    def __len__(self):
        return len(self.flist)