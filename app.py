from autogen.agentchat.contrib.teachable_agent import TeachableAgent
from autogen import UserProxyAgent
import openai


config_list = [
    {  
      "api_key":"sk-12345", 
      "api_base":"http://localhost:1234/v1",
      "api_type":"open_ai",
      
    }
]

openai.api_key = "sk-12345"
openai.api_base = "http://localhost:1234/v1"

llm_config = {
    "timeout": 60,
    "config_list": config_list,
    "use_cache": True,  # Use False to explore LLM non-determinism.
}

teach_config={
    "verbosity": 0,  # 0 for basic info, 1 to add memory operations, 2 for analyzer messages, 3 for memo lists.
    "reset_db": True,  # Set to True to start over with an empty database.
    "path_to_db_dir": "./tmp/notebook/teachable_agent_db",  # Path to the directory where the database will be stored.
    "recall_threshold": 1.5,  # Higher numbers allow more (but less relevant) memos to be recalled.
}

try:
    from termcolor import colored
except ImportError:
    def colored(x, *args, **kwargs):
        return x
    
teachable_agent = TeachableAgent(
    name="teachableagent",
    llm_config=llm_config,
    teach_config=teach_config)

user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
)

# Create UserProxyAgent
user = UserProxyAgent("user", human_input_mode="ALWAYS")

# Chat with TeachableAgent
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")

# Update the database
teachable_agent.learn_from_user_feedback()
teachable_agent.close_db()