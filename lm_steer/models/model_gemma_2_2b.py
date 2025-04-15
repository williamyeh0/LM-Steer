import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from .model_utils import Hack_no_grad, find_max_subspans
from .steers import Projected_Adaptor
from .model_base import LMSteerBase
from lm_steer.utils import set_seed

class Switching_Gemma2Model(LMSteerBase):
    def __init__(self, model_name, adapted_component, adaptor_class,
                 num_steers, rank, epsilon, init_var,
                 low_resource_mode):
        super().__init__()
        """ 
        model_name is directly provided in args
        you might need google/gemma-2-2b 
        """
        self.adapted_component = adapted_component
        if low_resource_mode:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name) # default?
            # below is from ProFS
            # model = AutoModelForCausalLM.from_pretrained(
            #     model_path,
            #     device_map="auto",
            #     # device_map={"": "cpu"}, # cpu lol
            #     torch_dtype=dtype,  # Non quantized weights are torch.float16 by default
            #     cache_dir=os.path.join(os.environ['HF_HOME'], 'hub'),
            #     token=hf_token,
            #     quantization_config=quantization_config
            #     # attn_implementation="eager" # gemma
            # )
        # print(f'model attributes in model_gemma_2_2b: {dir(self.model)}')
        # print(f'keys in model dict {self.model.__dict__.keys()}')

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # other params, cpu?
        self.init_var = init_var
        self.num_steers = num_steers
        self.device = torch.device("cuda:0")

        # embed_out?
        embed_dim = self.model.config.hidden_size
        vocab_size = self.model.config.vocab_size

        # print(f'embed_dim size: {embed_dim}')
        # print(f'vocab_size: {vocab_size}')
        self.low_resource_mode = low_resource_mode

        for _param in self.model.parameters():
            _param.requires_grad_(False)

        if adapted_component == "final_layer":
            self.model.model.layers = Hack_no_grad(self.model.model.layers)
            # devices?
            self.steer = Projected_Adaptor(
                self.model.lm_head,
                adaptor_class,
                num_steers,
                embed_dim,
                vocab_size,
                rank,
                epsilon,
                init_var,
                "output"
            )
            # self.steer = Projected_Adaptor(
            #     self.model.embed_out, adaptor_class, num_steers, embed_dim,
            #     vocab_size, rank, epsilon, init_var, "output")
            self.model.set_output_embeddings(self.steer)
        else:
            raise NotImplementedError()   
                 