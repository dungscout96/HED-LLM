from abc import ABC, abstractmethod
from transformers import GenerationConfig
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
from transformers import GPT2LMHeadModel
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
from transformers import BartForConditionalGeneration, BartTokenizer

def create_model(model_name):
    return globals()[model_name]()
    
class Model(ABC):
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.generation_config = GenerationConfig(max_new_tokens=55, do_sample=True)
        pass

    @property
    @abstractmethod
    def description(self):
        pass

    @abstractmethod
    def decode(self, input):
        print("Input", input)
        inputs = self.tokenizer(input, return_tensors='pt')
        output = self.tokenizer.decode(
            self.model.generate(
                inputs["input_ids"], 
                max_new_tokens=50,
                temperature=0.5
            )[0], 
            skip_special_tokens=True
        )
        return output

class FLANT5(Model):
    description = '''
FLAN-T5 was released in the paper Scaling Instruction-Finetuned Language Models - it is an enhanced version of T5 that has been finetuned in a mixture of tasks.
    '''

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
        
    def decode(self, input):
        return super().decode(input)
    
    
class GPT2(Model):
    description = '''
OpenAI GPT-2 model was proposed in Language Models are Unsupervised Multitask Learners by Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever from OpenAI.\n\
It’s a causal (unidirectional) transformer pretrained using language modeling on a very large corpus of ~40 GB of text data.
    '''
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def decode(self, input):
        return super().decode(input)

class BART(Model):
    description = '''
The Bart model was proposed in BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension by Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov and Luke Zettlemoyer on 29 Oct, 2019.
    '''
    def __init__(self):
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    def decode(self, input):
        batch = self.tokenizer(input, return_tensors="pt")
        generated_ids = self.model.generate(batch["input_ids"])
        output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return output

class GPTNeox(Model):
    description = '''
We introduce GPT-NeoX-20B, a 20 billion parameter autoregressive language model trained on the Pile, whose weights will be made freely and openly available to the public through a permissive license. It is, to the best of our knowledge, the largest dense autoregressive model that has publicly available weights at the time of submission. 
    '''
    def __init__(self):
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")
        self.model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b")

    def decode(self, input):
        input_ids = self.tokenizer(input, return_tensors="pt").input_ids
        gen_tokens = self.model.generate(
            input_ids,
            generation_config=self.generation_config
        )
        output = self.tokenizer.batch_decode(gen_tokens)[0]
        return output

