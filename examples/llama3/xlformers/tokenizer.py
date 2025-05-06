import os
from abc import ABC, abstractmethod

from logging import getLogger
from typing import AbstractSet, Collection, Iterator, List, Literal, Union

from sentencepiece import SentencePieceProcessor

from .args import TokenizerArgs
from .params import ConfStore

logger = getLogger()


class BaseTokenizer(ABC):
    def __init__(self, args: TokenizerArgs):
        assert os.path.exists(
            args.path
        ), f"The tokenizer path does not exist: {args.path}"
        self._bos_id = 0
        self._eos_id = 1
        self._pad_id = -1
        self._n_words = 2

    @property
    def bos_id(self) -> int:
        return self._bos_id

    @property
    def eos_id(self) -> int:
        return self._eos_id

    @property
    def pad_id(self) -> int:
        return self._pad_id

    @property
    def n_words(self) -> int:
        return self._n_words

    @abstractmethod
    def encode(self, *args, **kwargs) -> List[int]: ...

    @abstractmethod
    def decode(self, *args, **kwargs) -> str: ...


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, args: TokenizerArgs):
        super().__init__(args)
        self._tok_model = SentencePieceProcessor(model_file=args.path)
        logger.info(f"Reloaded SentencePiece model from {args.path}")
        assert self._tok_model.vocab_size() == self._tok_model.get_piece_size()

        self._bos_id = self._tok_model.bos_id()
        self._eos_id = self._tok_model.eos_id()
        self._pad_id = self._tok_model.pad_id()
        self._n_words = self._tok_model.vocab_size()

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def encode_str(self, s: str) -> List[str]:
        assert isinstance(s, str)
        return self._tok_model.encode(s, out_type=str)

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self._tok_model.encode(s)
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, tokens: List[int], cut_at_eos: bool = True) -> str:
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        return self._tok_model.decode(tokens)


def split_whitespaces_or_nonwhitespaces(
    s: str, max_consecutive_slice_len: int
) -> Iterator[str]:
    """
    Split the string `s` so that each substring contains no more than `max_consecutive_slice_len`
    consecutive whitespaces or consecutive non-whitespaces
    """
    current_slice_len = 0
    current_slice_is_space = s[0].isspace() if len(s) > 0 else False
    slice_start = 0

    for i in range(len(s)):
        is_now_space = s[i].isspace()

        if current_slice_is_space ^ is_now_space:
            current_slice_len = 1
            current_slice_is_space = is_now_space
        else:
            current_slice_len += 1
            if current_slice_len > max_consecutive_slice_len:
                yield s[slice_start:i]
                slice_start = i
                current_slice_len = 1
    yield s[slice_start:]


class TiktokenTokenizer(BaseTokenizer):
    BASIC_SPECIAL_TOKENS = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|fim_prefix|>",
        "<|fim_middle|>",
        "<|fim_suffix|>",
    ]

    LLAMA3_SPECIAL_TOKENS = (
        [
            "<|header_start|>",
            "<|header_end|>",
            "<|eom|>",
            "<|eot|>",
            "<|step_id|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|image|>",
            "<|video|>",
        ]
        + ["<|" + chr(ord("A") + i) + "|>" for i in range(26)]  # "<|A|>"..."<|Z|>"
        + ["<|" + chr(ord("0") + i) + "|>" for i in range(10)]  # "<|0|>"..."<|9|>"
    )

    CL100K_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

    def __init__(self, args: TokenizerArgs):
        super().__init__(args)
        try:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe

            mergeable_ranks = load_tiktoken_bpe(args.path)
        except ImportError:
            raise ImportError(
                "Please install tiktoken, blobfile and, lxml with `pip install tiktoken blobfile lxml`."
            )

        all_special_tokens_with_ids = self._get_all_special_tokens_with_ids(
            args, len(mergeable_ranks)
        )

        self._tok_model = tiktoken.Encoding(
            name=args.model,
            pat_str=args.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens={**all_special_tokens_with_ids},
        )
        logger.info(f"Reloaded Tiktoken model from {args.path}")

        self._bos_id = self.encode(
            TiktokenTokenizer.BASIC_SPECIAL_TOKENS[0],
            bos=False,
            eos=False,
            allowed_special="all",
        )[0]
        self._eos_id = self.encode(
            TiktokenTokenizer.BASIC_SPECIAL_TOKENS[1],
            bos=False,
            eos=False,
            allowed_special="all",
        )[0]
        self._pad_id = -1
        self._n_words = self._tok_model.n_vocab

        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )

    def _get_all_special_tokens_with_ids(
        self, args: TokenizerArgs, num_base_tokens: int
    ) -> dict:
        assert (
            len(args.basic_special_tokens) >= 2
        ), "Both bos and eos have to be specified in `TokenizerArgs.basic_special_tokens`."

        all_special_tokens = args.basic_special_tokens + args.extra_special_tokens

        assert len(set(all_special_tokens)) == len(
            all_special_tokens
        ), "Special tokens must be unique."

        n_vocab = num_base_tokens + args.num_reserved_special_tokens
        assert (
            n_vocab % 8 == 0
        ), "Vocabulary size must be divisible by 8 for vocabulary parallelism on 8 GPUs"

        assert (
            len(all_special_tokens) <= args.num_reserved_special_tokens
        ), "The total number of basic and extra special tokens exceeds the number of reserved tokens."

        reserved_tokens = [
            f"<|reserved_special_token_{i}|>"
            for i in range(args.num_reserved_special_tokens - len(all_special_tokens))
        ]
        all_special_tokens = (
            all_special_tokens[:-1] + reserved_tokens + [all_special_tokens[-1]]
        )

        return {
            token: num_base_tokens + i for i, token in enumerate(all_special_tokens)
        }

    def encode(
        self,
        s: str,
        bos: bool,
        eos: bool,
        allowed_special: Union[Literal["all"], AbstractSet[str]] = set(),
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[int]:
        assert isinstance(s, str)

        # The tiktoken tokenizer can handle <=400k chars without
        # pyo3_runtime.PanicException (may go beyond 400k)
        TIKTOKEN_MAX_ENCODE_CHARS = 400_000

        # Tiktoken is very bad at handling long sequences where either no whitespaces or only whitespaces:
        # https://github.com/openai/tiktoken/issues/195
        # Here we iterate over subsequences and split if we exceed the limit
        # of max consequtive non-whitespace or whitespace characters.
        MAX_NO_WHITESPACES_CHARS = 25_000

        # TODO check if MAX_NO_WHITESPACES_CHARS already fixes the issue with TIKTOKEN_MAX_ENCODE_CHARS

        substrs: List[str] = []
        t = []
        for i in range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS):
            substr = s[i : i + TIKTOKEN_MAX_ENCODE_CHARS]
            sliced_substr = split_whitespaces_or_nonwhitespaces(
                substr, MAX_NO_WHITESPACES_CHARS
            )
            substrs.extend(sliced_substr)
        for substr in substrs:
            # By default, setting disallowed_special=() encodes a string by
            # ignoring special tokens. Specifically:
            # - Setting `disallowed_special` to () will cause all text
            #   corresponding to special tokens to be encoded as natural
            #   text (insteading of raising an error).
            # - Setting `allowed_special` to "all" will treat all text
            #   corresponding to special tokens to be encoded as special tokens
            t.extend(
                self._tok_model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        if bos:
            t.insert(0, self.bos_id)
        if eos:
            t.append(self.eos_id)
        return t

    def decode(self, tokens: List[int], cut_at_eos: bool = True) -> str:
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break
        tokens = [token for token in tokens if token not in [self.bos_id, self.eos_id]]
        return self._tok_model.decode(tokens)


# For backward compatibility
class Tokenizer(BaseTokenizer):
    def __new__(cls, model_path=""):
        logger.warning(
            f"The `Tokenizer` class is deprecated and should not be used.\n"
            f"Please create a tokenizer with `core.data.tokenizers.build_tokenizer(tokenizer=tokenizer)`, "
            f"where `tokenizer` is among (register more at `core.data.tokenizers.py`):\n{REGISTERED_TOKS}\n"
            f"Alternatively, please use `SentencePieceTokenizer` or `TiktokenTokenizer` class instead."
        )

        if model_path.endswith(
            (
                "cl_toplang_128k",
                "cl_toplang_128k/",
                "cl100k_base",
                "cl100k_base/",
                "cl100k_base_96r",
                "cl100k_base_96r/",
            )
        ):
            tokenizer_args = ConfStore[f"{os.path.basename(model_path)}_pretrain_tok"]
            tokenizer_args.directory = os.path.dirname(model_path)
            return TiktokenTokenizer(tokenizer_args)

        args = TokenizerArgs(
            directory=os.path.dirname(model_path),
            model=os.path.basename(model_path),
        )
        return SentencePieceTokenizer(args)

    # Implementing abstract methods with no actual logic, since this class is deprecated
    # and should not be used for functionality
    def encode(self, *args, **kwargs) -> List[int]:
        logger.warning("The `Tokenizer` class is deprecated and should not be used.")
        return []

    def decode(self, *args, **kwargs) -> str:
        logger.warning("The `Tokenizer` class is deprecated and should not be used.")
        return ""


def build_tokenizer(tokenizer: TokenizerArgs) -> BaseTokenizer:
    # Tokenizer is registered in ConfStore
    if tokenizer.tokenizer_cls is not None:
        if isinstance(tokenizer.tokenizer_cls, str):
            # Dynamically retrieve the tokenizer class based on the class name string
            tokenizer_cls = globals()[tokenizer.tokenizer_cls]
        else:
            tokenizer_cls = tokenizer.tokenizer_cls
        return tokenizer_cls(tokenizer)
    return Tokenizer(model_path=tokenizer.path)


# Register tokenizers (ending with "_tok") in ConfStore

# cl_toplang_128k_pretrain_tok with Llama3 final special tokens
ConfStore["cl_toplang_128k_llama3_final_tok"] = TokenizerArgs(
    directory="/checkpoint/fair_llm/data/tokenizers/tiktoken",
    model="cl_toplang_128k",
    tokenizer_cls="TiktokenTokenizer",
    pat_str=TiktokenTokenizer.CL100K_PATTERN,
    num_reserved_special_tokens=256,
    basic_special_tokens=TiktokenTokenizer.BASIC_SPECIAL_TOKENS,
    extra_special_tokens=TiktokenTokenizer.LLAMA3_SPECIAL_TOKENS,
)


# Cl100k_base tokens + 28k top multilingual tokens = 128k tokens
ConfStore["cl_toplang_128k_pretrain_tok"] = TokenizerArgs(
    directory="/checkpoint/fair_llm/data/tokenizers/tiktoken",
    model="cl_toplang_128k",
    tokenizer_cls="TiktokenTokenizer",
    pat_str=TiktokenTokenizer.CL100K_PATTERN,
    num_reserved_special_tokens=256,
    basic_special_tokens=TiktokenTokenizer.BASIC_SPECIAL_TOKENS,
    extra_special_tokens=[],
)

# Cl100k_base tokenizer with 96 reserved special tokens, where vocab size
# 100256+96=100352, which is divisible by 2048: we can set model_parallelism as
# large as 2048, to use 2048/8=256 nodes (if we really want)
ConfStore["cl100k_base_96r_pretrain_tok"] = TokenizerArgs(
    directory="/checkpoint/fair_llm/data/tokenizers/tiktoken",
    model="cl100k_base_96r",
    tokenizer_cls="TiktokenTokenizer",
    pat_str=TiktokenTokenizer.CL100K_PATTERN,
    num_reserved_special_tokens=96,
    basic_special_tokens=TiktokenTokenizer.BASIC_SPECIAL_TOKENS,
    extra_special_tokens=[],
)

# Cl100k_base tokenizer with 256 reserved special tokens
ConfStore["cl100k_base_pretrain_tok"] = TokenizerArgs(
    directory="/checkpoint/fair_llm/data/tokenizers/tiktoken",
    model="cl100k_base",
    tokenizer_cls="TiktokenTokenizer",
    pat_str=TiktokenTokenizer.CL100K_PATTERN,
    num_reserved_special_tokens=256,
    basic_special_tokens=TiktokenTokenizer.BASIC_SPECIAL_TOKENS,
    extra_special_tokens=[],
)

# Experimental tokenizer
ConfStore["t4_128k_pretrain_tok"] = TokenizerArgs(
    directory="/checkpoint/fair_llm/data/tokenizers/T4",
    model="T4_128k_100T.tiktoken",
    tokenizer_cls="TiktokenTokenizer",
    pat_str=TiktokenTokenizer.CL100K_PATTERN,
    num_reserved_special_tokens=256,
    basic_special_tokens=TiktokenTokenizer.BASIC_SPECIAL_TOKENS,
    extra_special_tokens=[],
)

# Sentence Piece tokenizer for Llama 2 pretraining
ConfStore["llama2_pretrain_tok"] = TokenizerArgs(
    directory="/large_experiments/fair_llm/datasets/tokenizers",
    model="tokenizer_final_32k.minus_inf_ws.model",
    tokenizer_cls="SentencePieceTokenizer",
)

REGISTERED_TOKS = [s for s in ConfStore.keys() if s.endswith("_tok")]
