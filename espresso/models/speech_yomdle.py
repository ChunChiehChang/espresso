# Copyright (c) Yiming Wang
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderModel,
)
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.lstm import Linear
from fairseq.modules import FairseqDropout

from espresso.models.speech_lstm import ConvBNReLU,ConvBNReLU2
from espresso.models.speech_tdnn import TdnnBNReLU
import espresso.tools.utils as speech_utils


logger = logging.getLogger(__name__)


@register_model('speech_yomdle')
class SpeechYomdleModel(FairseqEncoderModel):
    def __init__(self, encoder, state_prior: Optional[torch.FloatTensor] = None):
        super().__init__(encoder)
        self.num_updates = 0
        self.state_prior = state_prior

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument("--dropout", type=float, metavar="D",
                            help="dropout probability")
        parser.add_argument("--residual", type=lambda x: options.eval_bool(x),
                            help="create residual connections for rnn encoder "
                            "layers (starting from the 2nd layer), i.e., the actual "
                            "output of such layer is the sum of its input and output")

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument("--dropout-in", type=float, metavar="D",
                            help="dropout probability for encoder\'s input")
        parser.add_argument("--dropout-out", type=float, metavar="D",
                            help="dropout probability for Tdnn layers\' output")
        parser.add_argument("--channels", type=int, default=1, metavar="INT",
                            help="the number of channels for color or grayscale images")
        # fmt: on

    @classmethod 
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted (in case there are any new ones)
        base_architecture(args)
        encoder = SpeechYomdleEncoder(
            task,
            feat_in_channels=task.feat_in_channels,
            dropout_in=args.dropout_in,
            dropout_out=args.dropout_out,
            residual=args.residual,
            chunk_width=getattr(task, "chunk_width", None),
            chunk_left_context=getattr(task, "chunk_left_context", 0),
            training_stage=getattr(task, "training_stage", True),
        )
        return cls(encoder, state_prior=getattr(task, "initial_state_prior", None))

    def set_num_updates(self, num_updates):
        self.num_updates = num_updates
        super().set_num_updates(num_updates)

    def output_lengths(self, in_lengths):
        return self.encoder.output_lengths(in_lengths)

    def get_normalized_probs(self, net_output, log_probs, sample=None): 
        """Get normalized probabilities (or log probs) from a net's output."""
        encoder_out = net_output.encoder_out
        if torch.is_tensor(encoder_out): 
            logits = encoder_out.float()
            if log_probs:
                return F.log_softmax(logits, dim=-1)
            else:
                return F.softmax(logits, dim=-1)
        raise NotImplementedError

    def get_logits(self, net_output):
        logits = net_output.encoder_out.transpose(0, 1).squeeze(2)  # T x B x 1 -> B x T
        return logits
        
    def update_state_prior(self, new_state_prior, factor=0.1):
        assert self.state_prior is not None
        self.state_prior = self.state_prior.to(new_state_prior)
        self.state_prior = (1. - factor) * self.state_prior + factor * new_state_prior
        self.state_prior = self.state_prior / self.state_prior.sum()  # re-normalize

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        state_dict["state_prior"] = self.state_prior
        return state_dict

    def load_state_dict(self, state_dict, strict=True, args=None):
        state_dict_subset = state_dict.copy()
        self.state_prior = state_dict.get("state_prior", None)
        if "state_prior" in state_dict:
            self.state_prior = state_dict["state_prior"]
            del state_dict_subset["state_prior"]
        super().load_state_dict(state_dict_subset, strict=strict, args=args)


class SpeechYomdleEncoder(FairseqEncoder):
    """Yomdle encoder."""
    def __init__(self, task, feat_in_channels=1, dropout_in=0.0, dropout_out=0.0, residual=False, chunk_width=None, chunk_left_context=0, training_stage=True):
        super().__init__(None)  # no src dictionary
        self.dropout_in_module = FairseqDropout(dropout_in, module_name=self.__class__.__name__)
        self.dropout_out_module = FairseqDropout(dropout_out, module_name=self.__class__.__name__)
        self.residual = residual
        #"""
        #tdnn_in_channels = int(512 * 60 * feat_in_channels / 4)
        tdnn_in_channels = 23040
        self.model = nn.ModuleList([
            ConvBNReLU2([32], [5], [1], in_channels=feat_in_channels),
            ConvBNReLU2([32], [5], [2], in_channels=32),
            ConvBNReLU2([128], [5], [1], in_channels=32),
            ConvBNReLU2([128], [5], [1], in_channels=128),
            ConvBNReLU2([128], [5], [2], in_channels=128),
            ConvBNReLU2([512], [3], [1], in_channels=128),
            ConvBNReLU2([512], [3], [1], in_channels=512),
            TdnnBNReLU(in_channels=tdnn_in_channels, out_channels=450, kernel_size=3, stride=1, dilation=4),
            TdnnBNReLU(in_channels=450, out_channels=450, kernel_size=3, stride=1, dilation=4),
            TdnnBNReLU(in_channels=450, out_channels=450, kernel_size=3, stride=1, dilation=4)
        ])
        # 512*60*3/4=23040
        # 128*60*3/2=11520

        receptive_field_radius = sum(layer.padding for layer in self.model)
        assert chunk_width is None or (chunk_width > 0 and chunk_left_context >= receptive_field_radius)
        if(
            chunk_width is not None and chunk_width > 0
            and chunk_left_context > receptive_field_radius
        ):
            logger.warning("chunk_{{left,right}}_context can be reduced to {}".format(receptive_field_radius))
        self.out_chunk_begin = self.output_lengths(chunk_left_context + 1) - 1
        self.out_chunk_end = self.output_lengths(chunk_left_context + chunk_width) \
            if chunk_width is not None else None
        self.training_stage = training_stage
        self.fc_out = Linear(450, task.num_targets)


    def output_lengths(self, in_lengths):
        out_lengths = in_lengths
        for layer in self.model:
            out_lengths = layer.output_lengths(out_lengths)
        return out_lengths

    def forward(self, src_tokens, src_lengths: Tensor, **unused):
        x, x_lengths, encoder_padding_mask = self.extract_features(src_tokens, src_lengths)
        if(
            self.out_chunk_end is not None
            and (self.training or not self.training_stage)
        ):
            # determine which output frame to select for loss evaluation/test, assuming
            # all examples in a batch are of the same length for chunk-wise training/test
            x = x[self.out_chunk_begin: self.out_chunk_end]  # T x B x C -> W x B x C

            x_lengths = x_lengths.fill_(x.size(0))
            assert not encoder_padding_mask.any()
        x = self.output_layer(x)

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask if encoder_padding_mask.any() else None,  # T x B
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=x_lengths,  # B
        )

    def extract_features(self, src_tokens, src_lengths, **unused):
        x, x_lengths = src_tokens, src_lengths
        x = self.dropout_in_module(x)

        for i in range(len(self.model)):
            if self.residual and i > 0:  # residual connection starts from the 2nd layer
                prev_x = x
            # apply Tdnn
            x, x_lengths, padding_mask = self.model[i](x, x_lengths)
            x = self.dropout_out_module(x)
            x = x + prev_x if self.residual and i > 0 and x.size(1) == prev_x.size(1) else x

        x = x.transpose(0, 1)  # B x T x C -> T x B x C
        encoder_padding_mask = padding_mask.t()

        return x, x_lengths, encoder_padding_mask

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        return self.fc_out(features)  # T x B x C -> T x B x V

    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        src_lengths: Optional[Tensor] = encoder_out.src_lengths
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(1, new_order)
        )
        new_src_lengths = (
            src_lengths
            if src_lengths is None
            else src_lengths.index_select(0, new_order)
        )
        return EncoderOut(
            encoder_out=encoder_out.encoder_out.index_select(1, new_order),
            encoder_padding_mask=new_encoder_padding_mask,
            encoder_embedding=None,
            encoder_states=None,
            src_tokens=None,
            src_lengths=new_src_lengths,
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number


@register_model_architecture("speech_yomdle", "speech_yomdle_color")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.dropout_in = getattr(args, "dropout_in", args.dropout)
    args.dropout_out = getattr(args, "dropout_out", args.dropout)
    args.residual = getattr(args, "residual", False)
    args.channels = getattr(args, "channels", 3)

@register_model_architecture("speech_yomdle", "speech_yomdle")
def base_architecture(args):
    args.dropout = getattr(args, "dropout", 0.1)
    args.dropout_in = getattr(args, "dropout_in", args.dropout)
    args.dropout_out = getattr(args, "dropout_out", args.dropout)
    args.residual = getattr(args, "residual", False)
