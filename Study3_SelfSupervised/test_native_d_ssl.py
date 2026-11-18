import unittest
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from native_d_ssl import NativeViews, WeightedDistributedDraws
from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
import torch


class NativeDTests(unittest.TestCase):
    def test_real_training_views_and_dynamic_mask(self):
        image=Image.fromarray(np.full((1936,2584,3),230,dtype=np.uint8))
        result=NativeViews()(image)
        self.assertEqual(len(result['local_crops']),8)
        batch=collate_data_and_cast([(result,()),(result,())],(.1,.5),.5,torch.float32,
                                   2304,MaskingGenerator((48,48),max_num_patches=1152))
        self.assertEqual(tuple(batch['collated_global_crops'].shape),(4,3,768,768))
        self.assertEqual(tuple(batch['collated_local_crops'].shape),(16,3,384,384))
        self.assertEqual(tuple(batch['collated_masks'].shape),(4,2304))

    def test_weighted_ddp_shards_and_epoch(self):
        weights=[0.,.2,.3,.5]*20
        full=WeightedDistributedDraws(weights,8,1,0)
        first=WeightedDistributedDraws(weights,8,2,0)
        second=WeightedDistributedDraws(weights,8,2,1)
        merged=[x for pair in zip(first,second) for x in pair]
        self.assertEqual(merged,list(full))
        self.assertFalse(any(i%4==0 for i in merged))
        first.set_epoch(1)
        self.assertNotEqual(list(first),merged[::2])


if __name__=='__main__':unittest.main()
