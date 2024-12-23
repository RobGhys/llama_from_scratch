from typing import Optional, List
import torch
import time
from pathlib import Path
import json

from llama_from_scratch.model import ModelArgs, Transformer
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

class Llama:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor,
                 model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args

    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str,
              load_model: bool, max_seq_len: int,
              max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, 'No checkpoints could be loaded. Not found.'
            check_path = checkpoints[0]
            print(f'Loading checkpoint from: {check_path=}')
            checkpoint = torch.load(check_path, map_location='cpu')
            print(f'Loaded checkpoint in: {(time.time() - prev_time):.2f} seconds')
            prev_time = time.time()
        with open(Path(checkpoints_dir) / 'params.json', 'r') as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(str(Path(checkpoints_dir) /tokenizer_path))

        model_args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.BFloat16Tensor)

        model = Transformer(model_args).to(device)

        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'Loaded state dict in: {(time.time() - prev_time):.2f}')

        return Llama(model, tokenizer, model_args)


    def text_completion(self, prompts: List[str], temperature: float = 0.6,
                        top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1

        # convert each prompt into tokens
        prompt_tokens = [self.tokenizer.encode(prompts, out_type=int, add_bos=True, add_eos=False) for prompt in prompts]

        # Verify batch size validity
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)

        # Create the list of the generated tokens, along with initial prompt tokens
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        for k, t in enumerate(prompt_tokens):
            # populate initial tokens with prompt tokens
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        eos_reached = torch.tensor([False] * batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id # true if the token is a prompt token, false otherwise
        for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos - 1: cur_pos])
            if temperature > 0:
                # apply temperature before softmax
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                # Greedy strategy: max prob token is selected
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # Only replace the token if it is a padding token
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token

            # EOS is reached only  if 'EOS' token found in padding position
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id())
            # EOS reached for all prompts
            if all(eos_reached):
                break

        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            # cut to the EOS token if present
            if self.tokenizer.eos_id() in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id())
                current_prompt_tokens = current_prompt_tokens[: eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))

        return (out_tokens, out_text)

    def sample_top_p(self, probs, p):
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort,dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs.sum(dim=-1, keepdim=True)) # redistribute probabilities among surviving tokens
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)

        return next_token


if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = False
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "coucou ma biche ça va ?"
    ]

    model = Llama.build(
        checkpoints_dir='../Llama-2-7b/',
        tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=1024,
        max_batch_size=len(prompts),
        device=device
    )

    print('successfully loaded the model.')

    out_tokens, out_text = (model.text_completion(prompts, max_gen_len=64))
    assert len(out_text) == len(prompts)

    for i in range(len(out_text)):
        print(f'{out_text[i]}')
        print(f'-' * 50)