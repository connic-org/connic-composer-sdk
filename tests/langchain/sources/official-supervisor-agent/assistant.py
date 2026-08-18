from langchain.agents import create_agent
from langchain.tools import tool


@tool
def create_calendar_event(title: str, start_time: str, end_time: str) -> str:
    """Create a calendar event."""
    return f"Event created: {title} from {start_time} to {end_time}"


@tool
def get_available_time_slots(date: str, duration_minutes: int) -> list[str]:
    """Check calendar availability for a given date."""
    return ["09:00", "14:00", "16:00"]


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email notification."""
    return f"Email sent to {to} with subject '{subject}'"


calendar_agent = create_agent(
    model="openai:gpt-5.2",
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=(
        "You are a calendar scheduling assistant. "
        "Parse scheduling requests and use the calendar tools to fulfill them."
    ),
)


email_agent = create_agent(
    model="openai:gpt-5.2",
    tools=[send_email],
    system_prompt=(
        "You are an email assistant. Compose professional emails and send them when needed."
    ),
)


@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language."""
    result = calendar_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language."""
    result = email_agent.invoke({"messages": [{"role": "user", "content": request}]})
    return result["messages"][-1].text


supervisor_agent = create_agent(
    model="openai:gpt-5.2",
    tools=[schedule_event, manage_email],
    system_prompt=(
        "You are a helpful personal assistant. "
        "Break down user requests into scheduling and email actions as needed."
    ),
)
