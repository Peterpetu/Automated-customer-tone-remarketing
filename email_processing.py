from langchain.agents.agent_toolkits import ZapierToolkit
from langchain.utilities.zapier import ZapierNLAWrapper
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from faker import Faker
import os

class EmailProcessing:
    def __init__(self):
        key = os.getenv("ZAPIER_NLA_API_KEY")
        zapier = ZapierNLAWrapper(zapier_nla_api_key=key)
        self.toolkit = ZapierToolkit.from_zapier_nla_wrapper(zapier)
        self.llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-16k", verbose=True)
        self.agent = initialize_agent(self.toolkit.get_tools(), self.llm, agent="zero-shot-react-description", verbose=True)
        self.faker = Faker()
        self.domain = '@YourLangchainTestDomain.com'

    def formulate_email(self, email, name, review, summary):
        q = f"""
        The customer {name} just gave the following review {review}
        Formulate and send an email to {email} based on the review that {name} gave
        and take into account the overall summary of the review given here: '{summary}'.
        If there is not enough information in the review summary, generate a valid response. 
        The email should be signed with the name Benjamin and generate email_body_HTML.
        """
        return q

    def send_emails(self, df, summary):
        df['first_name'] = df.apply(lambda row: self.faker.first_name(), axis=1)
        df['last_name'] = df.apply(lambda row: self.faker.last_name(), axis=1)
        df['email'] = df.apply(lambda row: row['first_name'].lower() + row['last_name'].lower() + self.domain, axis=1)
        df = df[['first_name', 'last_name', 'reviewText', 'email', 'overall']]
        df.apply(lambda row: self.agent.run(self.formulate_email(row['email'], row['first_name'], row['reviewText'], summary)), axis=1)
