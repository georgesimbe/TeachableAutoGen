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
        # Initialize other specialized agents here...

    def handle_query(self, user_input):
        # Example of routing to the FinancialAdvisor
        if "stock" in user_input.lower():
            response = self.financial_advisor.advise_on_finance(user_input)
            if response:
                # If visualization needed, call visualized_data
                return visualized_data(response)
            return response
        # Add routing for other specialized agents here...
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
        # Existing implementation...
        pass

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
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={query}&apikey={api_key}'
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else "Error fetching financial data"
    except Exception as e:
        return "Exception in fetching financial data"

def fetch_news_articles(topic):
    api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={topic}&apikey={api_key}'
    try:
        response = requests.get(url)
        return response.json() if response.status_code == 200 else "Error fetching news articles"
    except Exception as e:
        return "Exception in fetching news articles"

def summarize_and_validate(data):
    summary = mistral_llm(data)
    return validate_with_gpt4(summary)
def validate_with_gpt4(summary):
    try:
        response = openai.Completion.create(
            engine="gpt-4-1106-preview",
            prompt=f"Please review this summary for accuracy: '{summary}'",
            max_tokens=100,
            api_key=config['openai']['api_key']
        )
        review = response.choices[0].text.strip()
        logging.info(f"Summary review: {review}")
        return review if "satisfactory" in review.lower() else "Summary needs improvement."
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

# Chat with TeachableAgent
teachable_agent.initiate_chat(user, message="Hi, I'm a teachable user assistant! What's on your mind?")

# Update the database
teachable_agent.learn_from_user_feedback()
teachable_agent.close_db()


