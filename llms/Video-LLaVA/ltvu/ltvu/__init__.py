import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
from transformers import logging as hf_logging
hf_logging.set_verbosity_error()
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

from videollava.model import LlavaLlamaForCausalLM
VideoLLaVA = LlavaLlamaForCausalLM  # to prevent a circular import

from .utils.logconfig import setup_logger
logger = setup_logger(__name__)
