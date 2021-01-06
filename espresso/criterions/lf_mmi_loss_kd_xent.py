# Copyright (c) Yiming Wang, Yiwen Shao
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pdb

import logging
import math

import torch

from espresso.criterions.lf_mmi_loss import ChainLossFunction

from fairseq import checkpoint_utils, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.logging import metrics


logger = logging.getLogger(__name__)

@register_criterion("lattice_free_mmi_kd_xent")
class LatticeFreeMMICriterionKDXent(FairseqCriterion):

    def __init__(
        self, task, sentence_avg, denominator_fst_path,
        leaky_hmm_coefficient, xent_regularize, output_l2_regularize,
        teacher_model_path, kd_topk
    ):
        super().__init__(task)
        try:
            from pychain.graph import ChainGraph
            import simplefst
        except ImportError:
            raise ImportError(
                "Please install OpenFST and PyChain by `make openfst pychain` "
                "after entering espresso/tools"
            )

        self.sentence_avg = sentence_avg
        den_fst = simplefst.StdVectorFst.read(denominator_fst_path)
        self.den_graph = ChainGraph(den_fst, initial_mode="leaky", final_mode="ones")
        self.leaky_hmm_coefficient = leaky_hmm_coefficient
        self.xent_regularize = xent_regularize
        self.output_l2_regularize = output_l2_regularize
        self.teacher_model, self._teacher_model_args = checkpoint_utils.load_model_ensemble(utils.split_paths(teacher_model_path)) 
        self.teacher_model = self.teacher_model[0]
        self.teacher_model.to(torch.cuda.current_device())

        self.topk = kd_topk
        self.KDLoss = torch.nn.KLDivLoss(reduction='sum')

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        FairseqCriterion.add_args(parser)
        parser.add_argument("--denominator-fst-path", type=str, metavar="FILE",
                            help="path to the denominator fst file")
        parser.add_argument("--leaky-hmm-coefficient", default=1.0e-05, type=float, metavar="F",
                            help="leaky-hmm coefficient for the denominator")
        parser.add_argument("--xent-regularization-coefficient", default=0.0,
                            type=float, metavar="F", dest="xent_regularize",
                            help="cross-entropy regularization coefficient")
        parser.add_argument("--output-l2-regularization-coefficient", default=0.0,
                            type=float, metavar="F", dest="output_l2_regularize",
                            help="L2 regularization coefficient for the network's output")
        parser.add_argument("--teacher-model-path", type=str, metavar="FILE",
                            help="path to teacher model")
        parser.add_argument("--kd-topk", default=10, type=int, metavar="INT",
                            help="Compare the top K values for KD")
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        teacher_output = self.teacher_model(**sample["net_input"])
        loss, nll_loss, kd_loss = self.compute_loss(net_output, teacher_output, sample, reduce=reduce)

        sample_size = sample["target"].batch_size if self.sentence_avg else sample["ntokens"]
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "kd_loss": kd_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, net_output, teacher_output, sample, reduce=True):
        try:
            from pychain.graph import ChainGraphBatch
            from pychain.loss import ChainFunction
        except ImportError:
            raise ImportError("Please install OpenFST and PyChain by `make openfst pychain` after entering espresso/tools")

        encoder_out = net_output.encoder_out.transpose(0, 1)  # T x B x V -> B x T x V
        out_lengths = net_output.src_lengths.long()  # B
        den_graphs = ChainGraphBatch(self.den_graph, sample["nsentences"])
        if self.xent_regularize > 0.0:
            den_objf = ChainFunction.apply(encoder_out, out_lengths, den_graphs, self.leaky_hmm_coefficient)
            num_objf = ChainFunction.apply(encoder_out, out_lengths, sample["target"])
            loss = - num_objf + den_objf  # negative log-probs
            nll_loss = loss.clone().detach()
            loss -= self.xent_regularize * num_objf
        else:
            # demonstrate another more "integrated" usage of the PyChain loss. it's equivalent to
            # the first three lines in the above "if" block, but also supports throwing away
            # batches with the NaN loss by setting their gradients to 0.
            loss = ChainLossFunction.apply(
                encoder_out, out_lengths, sample["target"], den_graphs, self.leaky_hmm_coefficient
            )
            nll_loss = loss.clone().detach()

        if self.output_l2_regularize > 0.0:
            encoder_padding_mask = net_output.encoder_padding_mask
            encoder_out_squared = encoder_out.pow(2.0)
            if encoder_padding_mask is not None:
                pad_mask = encoder_padding_mask.transpose(0, 1).unsqueeze(-1)  # T x B -> B x T x 1
                encoder_out_squared.masked_fill_(pad_mask, 0.0)
            loss += 0.5 * self.output_l2_regularize * encoder_out_squared.sum()
        
        kd_loss = 100*self.compute_loss_kd(net_output, teacher_output)
        loss += kd_loss
        
        return loss, nll_loss, kd_loss

    def compute_loss_kd(self, net_output, teacher_output):
        values, indices = torch.topk(net_output.encoder_out, self.topk, dim=2)
        student_matrix = torch.zeros(net_output.encoder_out.shape).to(torch.cuda.current_device()).scatter(2, indices, values)
        values, indices = torch.topk(teacher_output.encoder_out, self.topk, dim=2)
        teacher_matrix = torch.zeros(teacher_output.encoder_out.shape).to(torch.cuda.current_device()).scatter(2, indices, values)
        loss = self.KDLoss(student_matrix.permute(1,2,0), teacher_matrix.permute(1,2,0))
        return loss

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        return state_dict
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        state_dict_subset = state_dict.copy()
        if "state_prior" in state_dict:
            del state_dict_subset["state_prior"]
        super().load_state_dict(state_dict_subset, *args, **kwargs)

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get('kd_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=7)
        metrics.log_scalar("nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=7)
        metrics.log_scalar("kd_loss", kd_loss_sum / sample_size, sample_size, round=7)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg, round=4))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
