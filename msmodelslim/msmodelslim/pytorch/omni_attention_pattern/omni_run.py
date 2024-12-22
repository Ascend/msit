# run this python file to find the pattern
import logging
from msmodelslim.pytorch.omni_attention_pattern.omni_config import OmniAttentionConfig
from msmodelslim.pytorch.omni_attention_pattern.omni_tools import OmniAttentionGeneticSearcher

# configure logging
logging.basicConfig(level=logging.INFO)
config = OmniAttentionConfig(debug=False, model_path="/home/ma-user/work/Qwen25-instruct-7B", pool_size=20)
logging.info("Config: %s", config)

searcher = OmniAttentionGeneticSearcher(config)
searcher.search()