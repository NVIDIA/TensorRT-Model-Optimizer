import copy
import gc
import logging
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from transformers import BatchEncoding

from modelopt.deploy.llm.generate import LLM

logger = logging.getLogger(__name__)

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]


@register_model("trt-llm")
class TRTLLM(TemplateAPI):
    def __init__(
        self,
        tokenizer: str,
        engine_dir: str,
        batch_size: int = 1,
        max_gen_toks: int = 256,
        **kwargs,
    ):
        assert isinstance(tokenizer, str)
        super().__init__(tokenizer=tokenizer, batch_size=int(batch_size), max_gen_toks=max_gen_toks)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        assert isinstance(engine_dir, str)

        self.llm = LLM(engine_dir=engine_dir, tokenizer=self.tokenizer)
        logger.info("Loaded TRT-LLM engine")

    def _generate(
        self,
        input_tokens,
        max_tokens: int,
        **generation_kwargs: dict,
    ) -> dict:
        with torch.no_grad():
            outputs = self.runner.generate(
                batch_input_ids=input_tokens,
                max_new_tokens=max_tokens,
                end_id=self.tokenizer.eos_token_id,
                pad_id=self.tokenizer.pad_token_id,
                return_dict=True,
                **generation_kwargs,
            )
        torch.cuda.synchronize()
        return outputs

    def model_call(
        self,
        messages: Iterable[List[int]],
        *,
        generate: bool = True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        # !!! Copy: shared dict for each request, need new object !!!
        gen_kwargs = copy.deepcopy(gen_kwargs)

        assert isinstance(messages, Iterable), "Expect the messages to be Iterable[List[int]]"
        first_element = next(iter(messages))
        assert isinstance(first_element, List) and isinstance(
            first_element[0], int
        ), "Expect the messages to be Iterable[List[int]]"

        if not generate:
            return self.llm.generate_context_logits(prompts=messages)

        llm_kwargs = {}
        if gen_kwargs:
            if "until" in gen_kwargs:
                llm_kwargs["stop_words"] = gen_kwargs.pop("until")
            if "temperature" in gen_kwargs:
                llm_kwargs["temperature"] = gen_kwargs.pop("temperature")
            if "top_p" in gen_kwargs:
                llm_kwargs["top_p"] = gen_kwargs.pop("top_p")

        output_texts = self.llm.generate_text(
            prompts=messages,
            max_new_tokens=self._max_gen_toks,
            keep_input_prompt=False,
            **llm_kwargs,
        )

        return output_texts

    async def amodel_call(
        self,
        session,
        messages: Iterable[List[int]],
        *,
        generate: bool = True,
        cache_keys: list = None,
        ctxlens: Optional[List[int]] = None,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ):
        raise NotImplementedError

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        *,
        generate: bool = True,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        **kwargs,
    ) -> dict:
        """This method is responsible for creating the json payload that will be sent to the API."""
        raise NotImplementedError

    @staticmethod
    def parse_generations(outputs: Union[Any, List[Any]], **kwargs) -> List[str]:
        """Method used to parse the generations from the (batched) API response."""
        return outputs

    @staticmethod
    def parse_logprobs(
        outputs: Union[Any, List[Any]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """Method used to parse the logprobs from the (batched) API response.

        The provided tokens have two parts: The context tokens (length as ctxlens) and the continuation tokens.
        The logprobs returned is computed from the continuation tokens.
        We return the sum of the logprob of the continuation tokens
            [assuming the continuation tokens are the golden output].
        """
        res = []

        for logits_single_batch, tokens_single_batch, ctxlen_single_batch in zip(
            outputs, tokens, ctxlens
        ):
            logits_single_batch = logits_single_batch.to("cuda")
            continuation_logprob = F.log_softmax(
                logits_single_batch[(ctxlen_single_batch - 1) : -1], dim=-1
            )
            continuation_tokens = torch.tensor(tokens_single_batch[ctxlen_single_batch:])
            top_tokens = continuation_logprob.argmax(dim=-1).cpu()

            is_greedy = torch.equal(top_tokens, continuation_tokens)

            logprob_sum = (
                continuation_logprob[
                    torch.arange(continuation_logprob.size(0)), continuation_tokens
                ]
                .sum()
                .cpu()
            )

            res.append((logprob_sum, is_greedy))

        return res


if __name__ == "__main__":
    cli_evaluate()
    # Force clean up the LLM instance and void hanging.
    gc.collect()
