"""Check global-only collation/loss and preserve the existing eight-local loss."""
import unittest
import torch
from train_dinov2_small_ssl import cross_view_dino_loss
from dinov2.data.collate import collate_data_and_cast
from dinov2.data.masking import MaskingGenerator
from dinov2.loss import DINOLoss


class GlobalOnlyTests(unittest.TestCase):
    def test_global_only_has_finite_gradients_and_no_local_batch(self):
        views={'global_crops':[torch.zeros(3,224,224),torch.ones(3,224,224)],'local_crops':[]}
        batch=collate_data_and_cast([(views,()),(views,())],(.1,.5),.5,torch.float32,
                                  n_tokens=196,mask_generator=MaskingGenerator((14,14),max_num_patches=98))
        self.assertEqual(batch['collated_global_crops'].shape[0],4)
        self.assertEqual(batch['collated_local_crops'].shape[0],0)
        global_logits=torch.randn(4,12,requires_grad=True)
        teacher=torch.randn(4,12).softmax(-1)
        loss=cross_view_dino_loss(DINOLoss(12),global_logits,None,teacher,0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(global_logits.grad).all())

    def test_eight_local_formula_is_unchanged(self):
        dino=DINOLoss(12)
        global_logits=torch.randn(4,12,requires_grad=True)
        local_logits=torch.randn(16,12,requires_grad=True)
        teacher=torch.randn(4,12).softmax(-1)
        s0,s1=global_logits.chunk(2);t0,t1=teacher.chunk(2)
        expected=(dino([s0],[t1])+dino([s1],[t0])+dino(local_logits.chunk(8),[t0,t1]))/18
        actual=cross_view_dino_loss(dino,global_logits,local_logits,teacher,8)
        torch.testing.assert_close(actual,expected)


if __name__=='__main__':
    unittest.main()
