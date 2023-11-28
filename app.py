import json
import os
import logging
from dotenv import load_dotenv
import requests
import openai
import matplotlib.pyplot as plt
import seaborn as sns

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
    @staticmethod
    def create_advisor(advisor_type):
        if advisor_type == "FinancialAdvisor":
            return FinancialAdvisor()
        elif advisor_type == "CryptoAdvisor":
            return CryptoAdvisor()
        elif advisor_type == "FinancialPlanner":
            return FinancialPlanner()
        elif advisor_type == "DebtRepairAdvisor":
            return DebtRepairAdvisor()
        else:
            raise ValueError("Invalid advisor type")

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
        }

    def handle_query(self, user_input):
        query_type = classify_query(user_input)
        response = None
        if query_type in self.advisors:
            for keyword, advisor in self.advisors[query_type].items():
                if keyword in user_input.lower():
                    response = advisor.advise(user_input)
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


    def learn_from_user_feedback(self):
        # Only learn from user feedback if not in production mode
        if not config['test_mode']:
            # Implement logic to learn from stored logs
            pass
        else:
            # In production, the agent should use its learned knowledge
            pass

    def respond_to_user(self, user_input):
        response = self.group_manager.handle_query(user_input)
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
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            elif response.status_code == 429:
                # Handle rate limit exceeded error
                time.sleep(1)  # Wait for a second before retrying
                return self.call_openai_api(user_input)
            else:
                return f"Error in API response: {response.status_code}, {response.text}"
        except Exception as e:
            return f"Exception in API call: {str(e)}"

class CryptoAdvisor:
    def validate_query(self, query):
        # Implement query validation logic here
        if not query:
            return "Insufficient data for advice."
        # Add more validation checks as needed
        return None

    def advise_on_crypto(self, query):
        validation_error = self.validate_query(query)
        if validation_error:
            return validation_error
        # Implement cryptocurrency advice logic here
        return "Cryptocurrency advice based on query"

class FinancialPlanner:
    def create_financial_plan(self, query):
        # Implement financial planning logic here
        return "Financial plan based on query"

class DebtRepairAdvisor:
    def provide_debt_repair_advice(self, query):
        # Implement debt repair advice logic here
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
def classify_query(user_input):
    financial_keywords = ["stock", "crypto", "market", "investment", "financial", "economy"]
    news_keywords = ["news", "headline", "current events", "article", "report"]
    if any(keyword in user_input.lower() for keyword in financial_keywords):
        return "finance"
    if any(keyword in user_input.lower() for keyword in news_keywords):
        return "news"
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

def summarize_and_validate(data):
    # Use Mistral model to summarize the data
    summary = mistral_llm(data)
    # Validate the summary using GPT-4
    validation = validate_with_gpt4(summary)
    # Check for factual accuracy and coherence in the validation
    if "accurate" in validation.lower() and "coherent" in validation.lower():
        return summary
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
    else:  # For other types, let's return a message
        return "Unable to visualize the given data."

    # Save the plot to a file
    fig.savefig('plot.png')
    return "A graph has been created and saved as plot.png"


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
        response = teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?", chat_history=chat_history)
        print(response)
        chat_history.append({"role": "assistant", "content": response})

    # Update the database
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()


