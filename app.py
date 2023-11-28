import json
import os
import logging
from dotenv import load_dotenv
import requests
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import time
import random
from autogen import UserProxyAgent, GPTAssistantAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from autogen.agentchat import AssistantAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configurations from environment and JSON
def load_configurations():
    """
    Load configurations from environment and JSON.
    Returns:
        dict: The configuration dictionary.
    """
    load_dotenv()
    with open('config.json') as config_file:
        config = json.load(config_file)

    openai_config = config.get('openai', {})
    config['openai'] = {
        'api_key': os.getenv('OPENAI_API_KEY', openai_config.get('api_key'))
    }

    config['test_mode'] = os.getenv('TEST_MODE', 'False').lower() == 'true'
    config['llm_config'] = config.get('llm_config', {})
    config['teach_config'] = config.get('teach_config', {})

    return config

config = load_configurations()



class AdvisorFactory:
    """
    Factory class for creating instances of specialized advisors.
    """
    advisor_cache = {}

    @staticmethod
    def create_advisor(advisor_type):
        if advisor_type in AdvisorFactory.advisor_cache:
            return AdvisorFactory.advisor_cache[advisor_type]

        if advisor_type == "FinancialAdvisor":
            advisor = FinancialAdvisor()
        elif advisor_type == "CryptoAdvisor":
            advisor = CryptoAdvisor()
        elif advisor_type == "FinancialPlanner":
            advisor = FinancialPlanner()
        elif advisor_type == "DebtRepairAdvisor":
            advisor = DebtRepairAdvisor()
        else:
            raise ValueError("Invalid advisor type")

        AdvisorFactory.advisor_cache[advisor_type] = advisor
        return advisor

class GroupManager:
    """
    Manages a group of specialized agents.
    """
    def __init__(self):
        """
        Initialize the GroupManager with a set of specialized agents.
        """
        self.advisors = {
            "finance": {
                "stock": AdvisorFactory.create_advisor("FinancialAdvisor"),
                "crypto": AdvisorFactory.create_advisor("CryptoAdvisor"),
                "plan": AdvisorFactory.create_advisor("FinancialPlanner"),
                "budget": AdvisorFactory.create_advisor("FinancialPlanner"),
                "debt": AdvisorFactory.create_advisor("DebtRepairAdvisor")
            }
            # Add other specialized agents here...
            # Add your agents here
        }

    def handle_query(self, user_input, chat_history):
        query_type = classify_query(user_input, chat_history)
        response = None
        if query_type in self.advisors:
            for keyword, advisor in self.advisors[query_type].items():
                if keyword in user_input.lower():
                    try:
                        response = advisor.advise(user_input)
                    except ValueError as e:
                        return f"Error: {str(e)}"
                    except Exception as e:
                        return f"Unexpected error: {str(e)}"
                    break
        # Add routing for other specialized agents here...

        if response:
            # If visualization needed, call visualized_data
            return visualized_data(response)
        elif response is None:
            return "No relevant data found."
        else:
            return "Error occurred while processing the query."
        return None

def exponential_backoff_retry(func, max_retries=5, max_delay=60):
    """
    Implements an exponential backoff retry strategy.
    Args:
        func (callable): The function to retry.
        max_retries (int): The maximum number of retries. Default is 5.
        max_delay (int): The maximum delay between retries in seconds. Default is 60.
    Returns:
        The result of the function call.
    """
    for n in range(max_retries):
        try:
            return func()
        except RateLimitExceededError:
            sleep_time = min((2 ** n) + random.random(), max_delay)
            time.sleep(sleep_time)
    raise Exception("Maximum retries exceeded")

class TeachableAgentWithLLMSelection:
    """
    A teachable agent that can select the appropriate language model based on the user's input.
    """
    def __init__(self, name, llm_config, teach_config, group_manager):
        """
        Initialize the TeachableAgentWithLLMSelection.
        Args:
            name (str): The name of the agent.
            llm_config (dict): The configuration for the language model.
            teach_config (dict): The configuration for the teaching process.
            group_manager (GroupManager): The group manager that manages the specialized agents.
        """
        self.name = name
        self.llm_config = llm_config
        self.teach_config = teach_config
        self.group_manager = group_manager
        self.api_key = config['openai']['api_key']

        # Create GPTAssistantAgent instances
        self.coder = GPTAssistantAgent("coder", llm_config=self.llm_config, instructions="You are a coder.")
        self.analyst = GPTAssistantAgent("analyst", llm_config=self.llm_config, instructions="You are an analyst.")


    def load_feedback_dataset():
        # Implement logic to load feedback data from a file or database
        pass

    def update_knowledge_base(feedback_dataset):
        # Implement logic to adjust the knowledge base or model parameters based on the feedback dataset
        pass

    def use_learned_knowledge():
        # Implement logic to apply the learned knowledge in production mode
        pass

    def learn_from_user_feedback(self):
        # Only learn from user feedback if not in production mode
        if not config['test_mode']:
            # Load feedback dataset
            feedback_dataset = self.load_feedback_dataset()
            # Update knowledge base or model parameters using feedback
            self.update_knowledge_base(feedback_dataset)
        else:
            # In production, the agent should use its learned knowledge
            self.use_learned_knowledge()

    def load_feedback_dataset(self):
        # Implement logic to load feedback data from a file or database
        pass

    def update_knowledge_base(self, feedback_dataset):
        # Implement logic to adjust the knowledge base or model parameters based on the feedback dataset
        pass

    def use_learned_knowledge(self):
        # Implement logic to apply the learned knowledge in production mode
        pass

    def respond_to_user(self, user_input, chat_history):
        response = self.group_manager.handle_query(user_input, chat_history)
        return response if response else "No relevant data found."

    def call_openai_api(self, user_input):
        # Implement rate limiting and retry logic to handle API limits
        # This is a placeholder for the rate limiting and retry mechanism
        # In a real-world scenario, this could involve a library like tenacity or retrying

        # Add detailed error handling for different HTTP status codes
        # This is a placeholder for the error handling mechanism
        # In a real-world scenario, this could involve a try/except block with different handlers for each status code

        # Ensure secure handling of API keys, possibly using environment variables or a secrets manager
        # This is a placeholder for the secure handling of API keys
        # In a real-world scenario, this could involve a secrets manager like AWS Secrets Manager or HashiCorp Vault

        messages = [{"role": "user", "content": user_input}]
        payload = {"model": "gpt-4-1106-preview", "messages": messages}
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            def api_call():
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                elif response.status_code == 429:
                    raise RateLimitExceededError("Rate limit exceeded")
                else:
                    raise APIError(f"Error in API response: {response.status_code}, {response.text}")
            return exponential_backoff_retry(api_call)
        except RateLimitExceededError as e:
            return f"Rate limit exceeded: {str(e)}"
        except APIError as e:
            return str(e)
        except Exception as e:
            logging.error(f"Unexpected error: {str(e)}", exc_info=True)
            return f"Unexpected error: {str(e)}"

class CryptoAdvisor:
    def validate_query(self, query):
        # Implement query validation logic here
        if not query:
            return "Insufficient data for advice."
        # Add more validation checks as needed
        # If the query doesn't meet the criteria, return a prompt asking for more specific information
        # Add your validation logic here
        return None

    def advise_on_crypto(self, query):
        validation_error = self.validate_query(query)
        if validation_error:
            return validation_error
        # Implement cryptocurrency advice logic here
        # Add your advice logic here
        return "Cryptocurrency advice based on query"

class FinancialPlanner:
    def create_financial_plan(self, query):
        # Implement financial planning logic here
        # Add your planning logic here
        return "Financial plan based on query"

class DebtRepairAdvisor:
    def provide_debt_repair_advice(self, query):
        # Implement debt repair advice logic here
        # Add your debt repair advice logic here
        return "Debt repair advice based on query"

# Example of a specialized agent
class FinancialAdvisor:
    def advise_on_finance(self, query):
        # Implement financial advice logic here
        return "Financial advice based on query"

    def advise_on_investment(self, query):
        # Implement investment advice logic here
        return "Investment advice based on query"

class GroupManager:
    # ... Existing __init__ method ...

    def handle_query(self, user_input):
        # ... Existing logic ...

        # Use specialized advisors based on the query
        if "crypto" in user_input.lower():
            advisor = CryptoAdvisor()
            return advisor.advise_on_crypto(user_input)
        elif "plan" in user_input.lower() or "budget" in user_input.lower():
            planner = FinancialPlanner()
            return planner.create_financial_plan(user_input)
        elif "debt" in user_input.lower():
            debt_repair = DebtRepairAdvisor()
            return debt_repair.provide_debt_repair_advice(user_input)
        # ... Additional conditions for other specialized agents ...



 #Function to interact with GPT-4 API
    def call_openai_api(self, user_input):
            messages = [{"role": "user", "content": user_input}]
            payload = {"model": "gpt-4-1106-preview", "messages": messages}
            headers = {"Authorization": f"Bearer {self.api_key}"}

            try:
                response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                return f"Error in API response: {response.status_code}, {response.text}"
            except Exception as e:
                return f"Exception in API call: {str(e)}"

# Define the functions for query classification, fetching financial data and news, summarizing and validating data
def classify_query(user_input, chat_history):
    financial_keywords = ["stock", "crypto", "market", "investment", "financial", "economy"]
    news_keywords = ["news", "headline", "current events", "article", "report"]
    if any(keyword in user_input.lower() for keyword in financial_keywords):
        return "finance"
    if any(keyword in user_input.lower() for keyword in news_keywords):
        return "news"
    # Use chat_history to maintain context throughout the conversation
    # Implement your logic here to provide context-aware responses
    return "general"

def fetch_financial_data(query):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Parse the query for more detailed input
    parsed_query = parse_query(query)
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={parsed_query}&apikey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            return "API limit exceeded"
        else:
            return f"Error fetching financial data: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def fetch_news_articles(topic):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    # Parse the topic for more detailed input
    parsed_topic = parse_topic(topic)
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={parsed_topic}&apikey={api_key}'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 429:
            return "API limit exceeded"
        else:
            return f"Error fetching news articles: {response.status_code}, {response.text}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def summarize_and_validate(data, user_input):
    # Use Mistral model to summarize the data
    summary = mistral_llm(data)
    # Validate the summary using GPT-4
    validation = validate_with_gpt4(summary)
    # Check for factual accuracy, coherence, and relevance in the validation
    if "accurate" in validation.lower() and "coherent" in validation.lower():
        # Check relevance of the summary to the user's query
        if user_input.lower() in summary.lower():
            return summary
        else:
            return "The summary was not relevant to your query."
    else:
        return "The summary was not accurate or coherent."
def validate_with_gpt4(summary):
    try:
        response = openai.Completion.create(
            engine="gpt-4-1106-preview",
            prompt=f"Please review this summary for accuracy and coherence: '{summary}'",
            max_tokens=100,
            api_key=config['openai']['api_key']
        )
        review = response.choices[0].text.strip()
        logging.info(f"Summary review: {review}")
        # Check for factual accuracy and coherence in the review
        if "accurate" in review.lower() and "coherent" in review.lower():
            return "The summary is accurate and coherent."
        else:
            return "The summary is not accurate or coherent."
    except Exception as e:
        logging.exception("Error calling GPT-4 API for summary validation")
        return "Unable to validate summary."


def mistral_llm(query):
    # Replace with your custom endpoint URL
    API_URL = "https://z983hozbx3in30io.us-east-1.aws.endpoints.huggingface.cloud"
    # Authorization token
    auth_token = "Bearer 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"
    # Prepare the request header
    headers = {
        "Authorization": auth_token,
        "Content-Type": "application/json"
    }
    # Prepare the request body
    data = json.dumps({"inputs": query})

    try:
        # Send POST request to the API
        response = requests.post(API_URL, headers=headers, data=data)
        if response.status_code == 200:
            return response.json()
        else:
            logging.error(f"API call failed: {response.status_code}, {response.text}")
            return None
    except Exception as e:
        logging.exception("Exception occurred during API call")
        return None


def visualized_data(dataset):
    # Determine the type of visualization
    if isinstance(dataset, dict):  # For simplicity, let's assume dict type data is for bar chart
        fig, ax = plt.subplots()
        sns.barplot(x=list(dataset.keys()), y=list(dataset.values()), ax=ax)
        ax.set_title('Bar Chart')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
    elif isinstance(dataset, list) and all(isinstance(i, tuple) and len(i) == 2 for i in dataset):  # For list of tuples, let's assume it's for line graph
        fig, ax = plt.subplots()
        sns.lineplot(x=[i[0] for i in dataset], y=[i[1] for i in dataset], ax=ax)
        ax.set_title('Line Graph')
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')
    # Add conditions for other visualization types like scatter plots, heatmaps, and pie charts
    else:  # For other types, let's return a message
        return "Unable to visualize the given data."

    # Save the plot to a file
    fig.savefig('plot.png')
    # Return the plot as a response
    return fig


# Main execution
if __name__ == "__main__":
    group_manager = GroupManager()
    teachable_agent = TeachableAgentWithLLMSelection(
        name="financial_teachable_agent",
        llm_config=config['llm_config'],
        teach_config=config['teach_config'],
        group_manager=group_manager
    )
    
    # Create UserProxyAgent
    user = UserProxyAgent("user", human_input_mode="ALWAYS")

    # Setup Group Chat
    groupchat = GroupChat(agents=[user, teachable_agent.coder, teachable_agent.analyst], messages=[], max_round=10)
    manager = GroupChatManager(groupchat)

    # Start a continuous interaction loop
    chat_history = []
    while True:
        # Get user input
        user_input = input("You: ")
        chat_history.append({"role": "user", "content": user_input})

        # Break the loop if the user types 'exit' or 'quit'
        if user_input.lower() in ['exit', 'quit']:
            break

        # Chat with TeachableAgent
        response = teachable_agent.respond_to_user(user_input, chat_history)
        print(response)
        chat_history.append({"role": "assistant", "content": response})

    # Update the database
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()


