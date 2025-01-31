"""Data handling methods."""

from curses.ascii import US
from typing import Any, Optional

import torch
from pydantic import BaseModel

USE_GPU = torch.cuda.is_available()


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0


class Batch(BaseModel):
    df_batch: Optional[Any] = None

    @property
    def win(self):
        window = self.df_batch.size(-2)
        return int((window - 1) / 2)

    @property
    def src(self):
        out = self.df_batch[:, : self.win, :]
        if USE_GPU:
            return out.to(0)
        else:
            return out

    @property
    def src_mask(self):
        return None

    @property
    def tgt(self):
        out = self.df_batch[:, self.win : -1, :]
        if USE_GPU:
            return out.to(0)
        else:
            return out

    @property
    def tgt_y(self):
        return self.df_batch[:, self.win + 1 :, :]

    @property
    def srctgt(self):
        out = self.df_batch[:, :-1, :]
        if USE_GPU:
            return out.to(0)
        else:
            return out

    @property
    def last_return(self):
        out = self.df_batch[:, -1, :]
        if USE_GPU:
            return out.to(0)
        else:
            return out

    @property
    def tgt_mask(self):
        tgt_mask1 = (
            torch.fill(self.tgt, 1).bool().unsqueeze(-3).transpose(2, 3).transpose(2, 1)
        ).to(0)
        tgt_mask2 = subsequent_mask(self.tgt.size(-2)).bool().to(0)
        return (tgt_mask1 & tgt_mask2)[:, 0, :, :]

    @property
    def ntokens(self):
        return (self.tgt_y).data.sum()
