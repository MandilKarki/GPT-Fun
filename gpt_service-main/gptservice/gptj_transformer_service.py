"""
-- Created by: Ashok Kumar Pant
-- Created on: 11/9/21
"""
import tensorflow as tf
from transformers import GPTJForCausalLM, AutoTokenizer


class GPTJGeneratorService:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B", revision="float16",
                                                     torch_dtype=tf.float16, low_cpu_mem_usage=True)

    def generate(self, prompt, do_sample=True, temperature=0.9, max_length=256):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(input_ids, do_sample=do_sample, temperature=temperature,
                                         max_length=max_length, )
        gen_text = self.tokenizer.batch_decode(gen_tokens)[0]
        return gen_text
