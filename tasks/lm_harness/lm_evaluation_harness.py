# Copyright (C) 2025 Intel Corporation

# This hack is a workaround to limitations of lm_eval which always allocates
# mp.Pool with max cpu count which explodes on multinode scenarios and for hpu
# create multiprocess with spawn context
import multiprocessing as mp
import psutil
import os

OrigPool = mp.Pool


def LimitedSpawnPool(_):
    spawn_context = mp.get_context("spawn")
    physical_cpu_count = psutil.cpu_count(logical=False)
    assert physical_cpu_count is not None
    world_size = int(os.getenv("WORLD_SIZE", 1))
    pool_size = max(1, physical_cpu_count // max(world_size, 1))
    if (pool_size * world_size) != physical_cpu_count:
        pool_size -= 1
    return spawn_context.Pool(pool_size)


mp.Pool = LimitedSpawnPool

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)

# flake8: noqa: E402
from functools import partial
from tqdm import tqdm
from typing import Iterable, List, Tuple

import torch
import torch.nn.functional as F

import lm_eval
from lm_eval import utils, tasks
from lm_eval.api.model import TemplateLM
from lm_eval.evaluator import evaluate

from pretrain_gpt import model_provider
from megatron.core import mpu
from megatron.core.models.gpt import GPTModel
from megatron.core.pipeline_parallel.schedules import send_forward, recv_forward
from megatron.training.checkpointing import load_checkpoint
from megatron.training import get_tokenizer, get_args, get_model
from megatron.training.initialize import initialize_megatron
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import get_ltor_masks_and_position_ids


class GPTModelWrapped(TemplateLM):
    def __init__(
        self,
        args,
        model: GPTModel,
        tokenizer: get_tokenizer,
        device="cuda",
        max_gen_tokens=128,
        batch_size=1,
        max_length=1024,
    ):
        super().__init__()
        self._model = model
        self._tokenizer = tokenizer
        self._device = device
        self._max_gen_tokens = max_gen_tokens
        self._batch_size = batch_size
        self._eot_token = self._tokenizer.eod
        self._prefix_token = (
            self._tokenizer.bos if "bos" in type(self._tokenizer).__dict__ else self._tokenizer.eod
        )
        self._pad_token = (
            self._tokenizer.bos if "pad" in type(self._tokenizer).__dict__ else self._tokenizer.eod
        )
        self._max_length = max_length
        self.config = core_transformer_config_from_args(args)
        self.args = args

    @property
    def eot_token_id(self):
        return self._eot_token

    @property
    def max_length(self):
        return self._max_length

    @property
    def max_gen_toks(self):
        return self._max_gen_tokens

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str):
        return self._tokenizer.tokenize(string)

    def tok_decode(self, tokens: Iterable[int]):
        return self._tokenizer.detokenize(tokens.tolist())

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation

        # automatic batch size detection for vectorization
        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            adaptive_batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {adaptive_batch_size}")

        loglikelihoods = []
        for string in tqdm(requests):
            rolling_token_windows = list(
                map(
                    utils.make_disjoint_window,
                    utils.get_rolling_token_windows(
                        token_list=self.tok_encode(string.doc['page']),
                        prefix_token=self._prefix_token,
                        max_seq_len=self.max_length,
                        context_len=1,
                    ),
                )
            )

            rolling_token_windows = [(None,) + x for x in rolling_token_windows]

            # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for
            # that
            string_nll = self._loglikelihood_tokens(
                rolling_token_windows, disable_tqdm=True, override_bs=adaptive_batch_size
            )

            # discard is_greedy
            string_nll = [x[0] for x in string_nll]

            string_nll = sum(string_nll)
            loglikelihoods.append(string_nll)

        return loglikelihoods

    def chunks(self, iterable, n=0, fn=None):
        arr = []
        for i, x in enumerate(iterable):
            arr.append(x)
            if len(arr) == (fn(i) if fn else n):
                yield arr
                arr = []

        if arr:
            yield arr

    def _loglikelihood_tokens(self, requests, disable_tqdm=False, override_bs=None):
        # TODO: implement some kind of efficient-request-middleware that lumps together requests with the same context
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end

            toks = x[1] + x[2]
            return -len(toks), tuple(toks)

        re_ord = utils.Reorderer(requests, _collate)
        reordered_requests = re_ord.get_reordered()
        n_reordered_requests = len(reordered_requests)

        # automatic (variable) batch size detection for vectorization
        # pull longest context sample from request
        def _batch_scheduler(pos):
            sched = pos // int(n_reordered_requests / self.batch_schedule)
            if sched in self.batch_sizes:
                return self.batch_sizes[sched]
            print(
                f"Passed argument batch_size = auto:{self.batch_schedule}. Detecting largest batch size"
            )
            self.batch_sizes[sched] = self._detect_batch_size(reordered_requests, pos)
            print(f"Determined largest batch size: {self.batch_sizes[sched]}")
            return self.batch_sizes[sched]

        for chunk in self.chunks(
            tqdm(reordered_requests, disable=disable_tqdm),
            n=(
                self.batch_size
                if self.batch_size != "auto"
                else override_bs if override_bs is not None else 0
            ),
            fn=(
                _batch_scheduler
                if self.batch_size == "auto" and n_reordered_requests > 0 and not override_bs
                else None
            ),
        ):
            inps = []
            cont_toks_list = []
            inplens = []

            padding_length = None

            # because vectorizing is annoying, we first convert each (context, continuation) pair to padded
            # tensors, then we pack them together into a batch, call the model, and then pick it all apart
            # again because vectorizing is annoying

            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works:
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # gpt2    \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                # when too long to fit in context, truncate from the left
                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                ).to(self.device)
                (inplen,) = inp.shape

                cont = continuation_enc

                # since in _collate we make sure length is descending, the longest is always the first one.
                padding_length = padding_length if padding_length is not None else inplen

                # pad length from seq to padding_length
                inp = torch.cat(
                    [
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(
                            inp.device
                        ),  # [padding_length - seq]
                    ],
                    dim=0,
                )

                inps.append(inp.unsqueeze(0))  # [1, padding_length]
                cont_toks_list.append(cont)
                inplens.append(inplen)

            batched_inps = torch.cat(inps, dim=0)  # [batch, padding_length]
            logits = self._model_call(batched_inps)

            # Logits is none in PP stages that are not last
            if logits is None:
                res.extend([(0.0, False)] * len(chunk))
                continue
            multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, padding_length, vocab]

            for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(
                chunk, multi_logits, inps, inplens, cont_toks_list
            ):

                # Slice to original seq length
                contlen = len(cont_toks)
                inplen = inplen + (
                    logits.shape[0] - padding_length
                )  # if "virtual tokens" (from prompt tuning) are added, inplen is larger
                logits = logits[inplen - contlen : inplen].unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)
                cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)  # [1, seq]
                max_equal = (greedy_tokens == cont_toks).all()

                # Obtain log-probs at the corresponding continuation token indices
                # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]

                # Answer: (log prob, is-exact-match)
                answer = (float(logits.sum()), bool(max_equal))

                # partial caching
                if cache_key is not None:
                    self.cache_hook.add_partial("loglikelihood", cache_key, answer)

                res.append(answer)

        return re_ord.get_original(res)

    def loglikelihood(self, requests, disable_tqdm: bool = False) -> List[Tuple[float, bool]]:
        new_reqs = []
        for context, continuation in [req.args for req in requests]:
            if context == "":
                # BOS or EOS as context
                context_enc, continuation_enc = (
                    [self.prefix_token_id],
                    self.tok_encode(continuation),
                )
            else:
                context_enc, continuation_enc = self._encode_pair(context, continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs, disable_tqdm=disable_tqdm)

    def generate_until(self, requests):
        raise RuntimeError('generate utils not implemented')

    def _model_call(self, inps):
        # Get the masks and postition ids.
        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            inps,
            self._tokenizer.eod,
            self.args.reset_position_ids,
            self.args.reset_attention_mask,
            self.args.eod_mask_loss,
        )

        batch_size, seq_len = inps.shape
        if mpu.is_pipeline_first_stage():
            # PP rank = 0 uses input inps
            input_tensor = inps
        else:
            # PP rank != 0 waits on output from previous PP rank for input
            input_tensor = recv_forward(
                [(inps.shape[0], inps.shape[1], self.args.hidden_size)], self.config
            )
            # len(models) = 1 => len(input_tensor) = 1
            self._model.set_input_tensor(input_tensor[0])

        # Forward pass. a_output.shape = [seq_len * batch, 1, vocab_size]
        a_output = self._model(input_tensor, position_ids.unsqueeze(dim=0), attention_mask)

        if not mpu.is_pipeline_last_stage():
            # If not last PP stage, forward output to next PP stage. Return nothing.
            send_forward(a_output, [a_output.shape], self.config)
            return None

        # Last PP stage returns logits
        if mpu.get_pipeline_model_parallel_world_size() > 1:
            return a_output.view(batch_size, seq_len, -1)
        return a_output


def get_lm_harness_args(parser):
    group = parser.add_argument_group(title='LM Eval Harness Arguments')

    group.add_argument(
        "--limit_iters", type=int, help="limit examples to run that many iterations", default=None
    )
    group.add_argument(
        "--max_new_tokens", type=int, default=500, help="Number of tokens to generate."
    )
    group.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        help="Tasks to run",
        default=["wikitext", "hellaswag", "lambada_openai", "piqa", "winogrande"],
    )

    return parser


def main():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['DISABLE_MULTIPROC'] = '1'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    initialize_megatron(extra_args_provider=get_lm_harness_args)

    args = get_args()
    assert args is not None
    if args.rank == 0:
        print(args)
    # Set up model and load checkpoint.
    model = get_model(partial(model_provider, parallel_output=False), wrap_with_ddp=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]
    model.eval()
    model.to(device)

    with torch.no_grad():
        results = evaluate(
            lm=GPTModelWrapped(
                model=model,
                tokenizer=get_tokenizer(),
                device=device,
                max_gen_tokens=args.max_new_tokens,
                max_length=4096,
                args=args,
            ),
            task_dict=tasks.get_task_dict(args.tasks),
            limit=args.limit_iters,
        )

    # Wait on all ranks before printing results. Avoids mix up with warnings.
    torch.distributed.barrier()
    if (
        mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_data_parallel_rank() == 0
        and mpu.is_pipeline_last_stage()
    ):
        print("Evaluation complete.")
        print(utils.make_table(results))


if __name__ == '__main__':
    main()
