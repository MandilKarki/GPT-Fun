"""
-- Created by: Ashok Kumar Pant
-- Created on: 9/16/21
"""

import logging
import threading
import time
from logging import INFO
from queue import Queue, Empty

import jax
import numpy as np
import optax
from curtsies.fmtfuncs import yellow, cyan
from jax.config import config
from jax.experimental import maps
from transformers import GPT2TokenizerFast

from gptservice import settings, helper
from gptservice.helper import timer
from gptservice.utils.cache import LRUCache
from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

logger = logging.getLogger(__name__)
logger.setLevel(INFO)


class GPTJModel:
    def __init__(self):
        self.params = {
            "layers": 28,
            "d_model": 4096,
            "n_heads": 16,
            "n_vocab": 50400,
            "norm": "layernorm",
            "pe": "rotary",
            "pe_rotary_dims": 64,
            "seq": 2048,
            "cores_per_replica": 8,
            "per_replica_batch": 1,
            "sampler": nucleaus_sample,
            "optimizer": optax.scale(0)
        }
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        self.queue_ids = {}
        self.qidx = 0
        self.queue = Queue()
        self.network = None
        self.model_path = None
        self.lock = threading.Lock()
        self._alive_time = timer()
        self.result_cache = LRUCache(1000)
        self.load_on_tup = False

    def prepare_tpu_device(self):
        # Not working with TPU-VM/TPU-Nodes, only for colab. This is not required for tpu-vm!
        logger.info("Preparing TPU device")
        google_tpu_addr = settings.TPU_NAME
        # url = f'http://{google_tpu_addr}/requestversion/tpu_driver0.1_dev20210607'
        # requests.post(url)

        # The following is required to use TPU Driver as JAX's backend.
        config.FLAGS.jax_xla_backend = "tpu_driver"
        config.FLAGS.jax_backend_target = "grpc://" + google_tpu_addr
        jax.default_backend()

    def load_model(self):
        if self.network:
            logger.info('Attempting to reload model when model is loaded. Returning')
            return
        with self.lock:
            logger.info("Getting model file")
            self.model_path = helper.download_or_get_gptj_6b_model()
            logger.info("Model path: {}".format(self.model_path))
            if self.model_path is None:
                return
            logger.info('Loading Model')
            start = timer()
            if self.load_on_tup:
                self.prepare_tpu_device()
            logger.info(f"JAX Devices: {jax.device_count()}")
            logger.info(f"JAX Runtime Initialized in {timer(start):.06} secs")
            mesh_shape = (jax.device_count() // self.params['cores_per_replica'], self.params['cores_per_replica'])
            self.devices = np.array(jax.devices()).reshape(mesh_shape)
            self.total_batch = self.params['per_replica_batch'] * jax.device_count() // self.params[
                'cores_per_replica'] * 8
            maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(self.devices, ('dp', 'mp')))
            network = CausalTransformer(self.params)
            logger.info(f'Loading Checkpoint')
            network.state = read_ckpt(network.state, self.model_path + "/", self.devices.shape[1])
            logger.info(f"GPTJ Network loaded in {timer(start):.06} secs. Total Batch Size: {self.total_batch}")
            del network.state["opt_state"]
            network.state = network.move_xmap(network.state, np.zeros(self.params['cores_per_replica']))
            self.network = network

    @property
    def is_model_loaded(self):
        return self.network is not None

    def start_background(self):
        with self.lock:
            t = threading.Thread(target=self.background)
            t.start()

    def prepare_item(self, context, length=256):
        tokens = self.tokenizer.encode(context)
        logger.info(tokens)
        token_length = len(tokens)
        pad_amount = self.params['seq'] - token_length
        pad_amount = max(pad_amount, 0)
        padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)[-self.params['seq']:]
        return {'tokens': padded_tokens, 'length': token_length}

    # Single Item - Not Tested #TODO Update as per infer_batch args
    def infer(self, context, top_p=0.9, top_k=40, temp=1.0, length=256, stop_sequence=None, next_line_only=False,
              **kwargs):
        item = self.prepare_item(context, length)
        batched_tokens = np.array([item['tokens']] * self.total_batch)
        batched_lengths = np.array([item['length']] * self.total_batch)
        start = timer()
        output = self.network.generate(
            batched_tokens, batched_lengths, length,
            {
                "top_p": np.ones(self.total_batch) * top_p,
                "top_k": np.ones(self.total_batch) * top_k,
                "temp": np.ones(self.total_batch) * temp,
            }
        )
        samples = []
        decoded_tokens = output[1][0]
        results = []
        for pred in decoded_tokens[:, :, 0]:
            results.append(self.tokenizer.decode(pred))

        end_time = timer(start)
        for text in results:
            text = text.split("<|endoftext|>")[0]
            # A simple technique to stop at stop_sequence without modifying the underlying model
            if stop_sequence is not None and stop_sequence in text:
                text = text.split(stop_sequence)[0] + stop_sequence
            if next_line_only:
                text = text.split('\n')[0]
            res = {
                'context': context,
                'text': text,
                'time': end_time,
                'stop_sequence': stop_sequence,
                'next_line_only': next_line_only
            }
            samples.append(res)
        logger.info(f"Completion done in {end_time:06} secs")
        return samples

    def infer_batch(self, batch, **kwargs):
        logger.info(f'Starting Inference on Batch')
        batch_items = {'tokens': [], 'lengths': [], 'top_p': [], 'top_k': [], 'temp': []}
        max_lengths, contexts = [], []
        stop_sequences = []
        next_line_only_list = []
        for req in batch:
            req = self.to_data(req)
            item = self.prepare_item(req['context'], req['length'])
            batch_items['tokens'].append(item['tokens'])
            batch_items['lengths'].append(item['length'])
            batch_items['top_p'].append(req['top_p'])
            batch_items['top_k'].append(req['top_k'])
            batch_items['temp'].append(req['temp'])
            stop_sequences.append(req['stop_sequence'])
            next_line_only_list.append(req['next_line_only'])
            max_lengths.append(req['length'])
            contexts.append(req['context'])

        max_length = max(max_lengths)
        for key, vals in batch_items.items():
            batch_items[key] = np.array(vals)
        start = timer()
        logger.info(f'Completed Preparing Batch')
        output = self.network.generate(
            batch_items['tokens'], batch_items['lengths'], max_length,
            {
                "top_p": batch_items['top_p'],
                "top_k": batch_items['top_k'],
                "temp": batch_items['temp'],
            }
        )
        logger.info(f'Completed Generation')
        results = []
        for pred in output[1][0][:, :, 0]:
            results.append(self.tokenizer.decode(pred))

        samples = []
        end_time = timer(start)
        for result, ctx, stop_sequence, next_line_only in zip(results, contexts, stop_sequences, next_line_only_list):

            text = result.split("<|endoftext|>")[0]

            # A simple technique to stop at stop_sequence without modifying the underlying model
            if stop_sequence is not None and stop_sequence in text:
                text = text.split(stop_sequence)[0] + stop_sequence
            if next_line_only:
                text = text.split('\n')[0]
            res = {
                'context': ctx,
                'text': text,
                'time': end_time,
                'stop_sequence': stop_sequence,
                'next_line_only': next_line_only
            }
            samples.append(res)
        logger.info(f"Completion done in {end_time:06} secs")
        return samples

    def add_to_queue(self, item):
        self.qidx += 1
        self.queue.put({'item': self.to_data(item), 'qidx': self.qidx})
        self.queue_ids[self.qidx] = Queue()
        return {'qid': self.qidx}

    def wait_for_queue(self, qid):
        if not self.queue_ids.get(qid):
            return {}
        result = self.result_cache.get(qid)
        if result is None:
            result = self.queue_ids[qid].get()
            self.result_cache.put(qid, result)
        return result

    def background(self):
        logger.info(f'Init Background')
        maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(self.devices, ('dp', 'mp')))
        while True:
            batch, qids = [], []
            while len(batch) <= self.total_batch:
                try:
                    req = self.queue.get(block=False)
                    logger.info(f'Got Queue Item: {req}')
                    batch.append(req['item'])
                    qids.append(req['qidx'])

                except Empty:
                    if len(batch):
                        break
                    else:
                        time.sleep(0.01)
            batch_size = len(batch)
            logger.info(f'Working on Batch: {batch_size} - {qids}')
            while len(batch) < self.total_batch:
                batch.append(self.placeholder_item)
            start = timer()
            results = self.infer_batch(batch)
            for res, qid in zip(results, qids):
                self.queue_ids[qid].put(res)
            logger.info(f'Completed Current Batch of {batch_size} Items in {timer(start):.2f} secs')

    @property
    def placeholder_item(self):
        return {'context': 'nada', 'top_p': 0.9, 'top_k': 40, 'temp': 1.0, 'length': 1}

    def to_data(self, item):
        try:
            return {'context': item.context, 'top_p': item.top_p, 'top_k': item.top_k, 'temp': item.temp,
                    'length': item.length, 'stop_sequence': item.stop_sequence, "next_line_only": item.next_line_only}
        except:
            return {'context': item.get('context', ''), 'top_p': item.get('top_p', 0.9), 'top_k': item.get('top_k', 40),
                    'temp': item.get('temp', 1.0), 'length': item.get('length', 256),
                    'stop_sequence': item.get('stop_sequence', None),
                    'next_line_only': item.get('next_line_only', False)}

    @property
    def alive_time(self):
        return timer(self._alive_time)

    def colorful_infer(self, model_in, gen_len=512, temp=1.0, top_p=0.9, top_k=40, next_line_only=False):
        inf = self.infer(model_in, gen_len=gen_len, temp=temp, top_p=top_p, top_k=top_k, next_line_only=next_line_only)
        print(yellow(model_in) + cyan(inf[0]))


# This prevents fastapi from creating multiple models in multi-worker mode
# leading to OOM crashes
_gptj_model: GPTJModel = None
_gptj_model_lock = threading.Lock()


def _compile_model():
    global _gptj_model
    with _gptj_model_lock:
        if _gptj_model:
            return
        _gptj_model = GPTJModel()


def get_gptj_model():
    _compile_model()
    return _gptj_model


if __name__ == '__main__':
    service = get_gptj_model()
    service.load_model()
    if not service.is_model_loaded:
        print("Model is not loaded")
        exit(0)
    while True:
        print("--------------------------------")
        context = input("Input: ")
        service.colorful_infer(context, gen_len=512)
