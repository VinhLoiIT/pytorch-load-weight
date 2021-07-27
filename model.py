from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from ocrstack.config.config import Config
from ocrstack.data.collate import Batch
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence

from resnet import resnet34


class TransformerDecoderAdapter(nn.Module):

    '''
    This class adapts `nn.TransformerDecoder` class to the stack
    '''

    def __init__(self):
        super(TransformerDecoderAdapter, self).__init__()
        self.in_embed, self.out_embed = self.build_embedding()
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(128, 8), 1)
        self.sos_idx = 0
        self.eos_idx = 1

    def build_embedding(self) -> Tuple[nn.Module, nn.Module]:
        out_embed = nn.Linear(128, 114, bias=False)
        in_embed = nn.Embedding(114, 128, 2,_weight=out_embed.weight)
        return in_embed, out_embed

    def forward(self, memory, tgt, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        '''
        Arguments:
        ----------
        - memory: (B, S, E)
        - tgt: (B, T)

        Returns:
        --------
        - logits: (B, T, V)
        '''
        # Since transformer components working with time-first tensor, we should transpose the shape first
        tgt = self.in_embed(tgt)                    # [B, T, E]
        tgt = tgt.transpose(0, 1)                   # [T, B, E]

        memory = memory.transpose(0, 1)             # [S, B, E]
        tgt_mask = generate_square_subsequent_mask(tgt.size(0)).to(memory.device)
        memory_mask = None
        output = self.decoder(tgt, memory, tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask)
        output = output.transpose(0, 1)                 # [B, T, E]
        output = self.out_embed(output)                 # [B, T, V]
        return output

    @torch.jit.export
    def decode(self, memory, max_length, memory_key_padding_mask=None):
        # type: (Tensor, int, Optional[Tensor]) -> Tensor
        batch_size = memory.size(0)
        inputs = torch.empty(batch_size, 1, dtype=torch.long, device=memory.device).fill_(self.sos_idx)
        outputs: List[Tensor] = [
            F.one_hot(inputs, num_classes=self.in_embed.num_embeddings).float().to(inputs.device)
        ]
        end_flag = torch.zeros(batch_size, dtype=torch.bool)
        for _ in range(max_length):
            text = self.forward(memory, inputs, memory_key_padding_mask, None)  # [B, T, V]
            output = F.softmax(text[:, [-1]], dim=-1)                           # [B, 1, V]
            outputs.append(output)                                              # [[B, 1, V]]
            output = output.argmax(-1, keepdim=False)                           # [B, 1]
            inputs = torch.cat((inputs, output), dim=1)                         # [B, T + 1]

            # set flag for early break
            output = output.squeeze(1)               # [B]
            current_end = output == self.eos_idx     # [B]
            current_end = current_end.cpu()
            end_flag |= current_end
            if end_flag.all():
                break

        return torch.cat(outputs, dim=1)                                   # [B, T, V]


class GeneralizedConvSeq2Seq(nn.Module):

    def __init__(self):
        # type: (Config,) -> None
        super().__init__()
        self.backbone = resnet34(pretrained=False, num_layers=2)
        self.decoder = TransformerDecoderAdapter()
        self.max_length = 150

    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)

    def predict(self, batch: Batch):
        predicts = self.forward(batch.images)
        return predicts

    def train_batch(self, batch: Batch):
        logits = self.forward(batch.images, batch.text, batch.lengths)
        return logits

    def compute_loss(self, logits, targets, lengths):
        packed_predicts = pack_padded_sequence(logits, lengths, batch_first=True)[0]
        packed_targets = pack_padded_sequence(targets, lengths, batch_first=True)[0]
        loss = F.cross_entropy(packed_predicts, packed_targets)
        return loss

    def example_inputs(self):
        return (torch.rand(1, 3, 64, 256), )

    def forward(self, images, text=None, lengths=None, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        images = self.backbone(images)                              # B, C, H, W

        B, E, H, W = images.shape
        images = images.reshape(B, E, H * W)                    # B, E, H * W
        images = images.transpose(-2, -1)                       # B, S = H * W, E

        if image_padding_mask is not None:
            image_padding_mask = image_padding_mask.reshape(B, H * W)

        if self.training:
            return self._forward_training(images, text, lengths, image_padding_mask)
        else:
            return self._forward_eval(images, image_padding_mask)

    @torch.jit.unused
    def _forward_training(self, images, text=None, lengths=None, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        text_padding_mask = generate_padding_mask_from_lengths(lengths - 1).to(images.device)      # B, S
        logits = self.decoder(images, text[:, :-1],
                              memory_key_padding_mask=image_padding_mask,
                              tgt_key_padding_mask=text_padding_mask)
        loss = self.compute_loss(logits, text[:, 1:], lengths - 1)
        return loss

    def _forward_eval(self, images, image_padding_mask=None):
        # type: (Tensor, Optional[Tensor]) -> Tensor
        predicts = self.decoder.decode(images, self.max_length, image_padding_mask)
        return predicts


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_padding_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    B, S = len(lengths), lengths.max()
    padding_mask = torch.arange(0, S, device=lengths.device).expand(B, S) >= lengths.unsqueeze(-1)
    return padding_mask
