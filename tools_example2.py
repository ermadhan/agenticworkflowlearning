# ================================
# Imports
# ================================

# --- Third-party ---
from dotenv import load_dotenv
import aisuite as ai
import json

# --- Local / project ---
import utils_tools2
import display_functions_tools2
import email_tools



# ================================
# Environment & Client
# ================================
load_dotenv()          # Load environment variables from .env
client = utils_tools2._get_openai_client()

# uncomment the line 'utils.test_*' you want to try
new_email_id = utils_tools2.test_send_email()
_ = utils_tools2.test_get_email(new_email_id['id'])
#_ = utils_tools2.test_list_emails()
#_ = utils_tools2.test_filter_emails(recipient="test@example.com")
#_ = utils_tools2.test_search_emails("lunch")
#_ = utils_tools2.test_unread_emails()
#_ = utils_tools2.test_mark_read(new_email_id['id'])
#_ = utils_tools2.test_mark_unread(new_email_id['id'])
#_ = utils_tools2.test_delete_email(new_email_id['id'])
#_ = utils_tools2.reset_database()

# Test sending a new email and fetch it by ID
new_email = email_tools.send_email("test@example.com", "Lunch plans", "Shall we meet at noon?")
content_ = email_tools.get_email(new_email['id'])

# Uncomment the ones you want to try:
#content_ = email_tools.list_all_emails()
#content_ = email_tools.list_unread_emails()
#content_ = email_tools.search_emails("lunch")
#content_ = email_tools.filter_emails(recipient="test@example.com")
#content_ = email_tools.mark_email_as_read(new_email['id'])
#content_ = email_tools.mark_email_as_unread(new_email['id'])
#content_ = email_tools.search_unread_from_sender("test@example.com")
#content_ = email_tools.delete_email(new_email['id'])

utils_tools2.print_html(content=json.dumps(content_, indent=2), title="Testing the email_tools")

def build_prompt(request_: str) -> str:
    return f"""
- You are an AI assistant specialized in managing emails.
- You can perform various actions such as listing, searching, filtering, and manipulating emails.
- Use the provided tools to interact with the email system.
- Never ask the user for confirmation before performing an action.
- If needed, my email address is "you@email.com" so you can use it to send emails or perform actions related to my account.

{request_.strip()}
"""

example_prompt = build_prompt("Delete the Happy Hour email")
utils_tools2.print_html(content=example_prompt, title="Example example_prompt")

utils_tools2.reset_database()

# Try your own requests
prompt_ = build_prompt("Check for unread emails from boss@email.com, mark them as read, and send a polite follow-up.")

response = utils_tools2.run_email_agent(client, prompt_, model="gpt-4.1")
print(response)

# Try with a request that may call an unavailable tool
prompt_ = build_prompt("Delete alice@work.com email")

response = utils_tools2.run_email_agent(client, prompt_, model="gpt-5-mini")
print(response)

prompt_ = build_prompt("Delete alice@work.com email")

response = utils_tools2.run_email_agent(client, prompt_, model="gpt-5-mini", include_delete=True)
print(response)

prompt_ = build_prompt("Delete the happy hour email")

response = utils_tools2.run_email_agent(client, prompt_, model="gpt-5-mini", include_delete=True)
print(response)