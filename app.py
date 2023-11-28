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



class GroupManager:
    def __init__(self):
        self.financial_advisor = FinancialAdvisor()
        self.crypto_advisor = CryptoAdvisor()
        self.financial_planner = FinancialPlanner()
        self.debt_repair_advisor = DebtRepairAdvisor()
        # Initialize other specialized agents here...

    def handle_query(self, user_input):
        query_type = classify_query(user_input)
        response = None
        if query_type == "finance":
            if "stock" in user_input.lower():
                response = self.financial_advisor.advise_on_finance(user_input)
            elif "crypto" in user_input.lower():
                response = self.crypto_advisor.advise_on_crypto(user_input)
            elif "plan" in user_input.lower() or "budget" in user_input.lower():
                response = self.financial_planner.create_financial_plan(user_input)
            elif "debt" in user_input.lower():
                response = self.debt_repair_advisor.provide_debt_repair_advice(user_input)
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
    def __init__(self, name, llm_config, teach_config, group_manager):
        self.name = name
        self.llm_config = llm_config
        self.teach_config = teach_config
        self.group_manager = group_manager
        self.api_key = config['openai']['api_key']


    def learn_from_user_feedback(self):
        # Only learn from user feedback if not in production mode
        if not IS_PROD:
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
    def advise_on_crypto(self, query):
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
        ax.bar(dataset.keys(), dataset.values())
        ax.set_title('Bar Chart')
        ax.set_xlabel('Categories')
        ax.set_ylabel('Values')
    else:  # For other types, let's assume it's for line graph
        fig, ax = plt.subplots()
        ax.plot(dataset)
        ax.set_title('Line Graph')
        ax.set_xlabel('Time')
        ax.set_ylabel('Values')

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
    while True:
        # Chat with TeachableAgent
        response = teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")
        print(response)
        
        # Get user input
        user_input = input("You: ")
        
        # Break the loop if the user types 'exit' or 'quit'
        if user_input.lower() in ['exit', 'quit']:
            break

    # Update the database
    teachable_agent.learn_from_user_feedback()
    teachable_agent.close_db()


