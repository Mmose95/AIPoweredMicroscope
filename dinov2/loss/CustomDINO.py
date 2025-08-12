import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomDINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.out_dim = out_dim
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.warmup_teacher_temp = warmup_teacher_temp
        self.warmup_teacher_temp_epochs = warmup_teacher_temp_epochs
        self.nepochs = nepochs
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.teacher_temp_schedule = torch.linspace(
            warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs
        )
        if warmup_teacher_temp_epochs < nepochs:
            self.teacher_temp_schedule = torch.cat((
                self.teacher_temp_schedule,
                torch.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
            ))

    def forward(self, student_output_chunks, teacher_output_chunks):
        student_out = torch.stack(student_output_chunks)  # (ncrops, batch, dim)
        teacher_out = torch.stack(teacher_output_chunks)  # (2, batch, dim), typically 2 global views

        # Compute teacher probabilities
        teacher_out = F.softmax((teacher_out - self.center) / self.teacher_temp_schedule[-1], dim=-1)
        teacher_out = teacher_out.detach()

        total_loss = 0
        n_loss_terms = 0

        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v] / self.student_temp, dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1

        total_loss /= n_loss_terms

        # Update center
        self.update_center(teacher_out)

        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_out):
        batch_center = torch.mean(teacher_out, dim=(0, 1), keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)