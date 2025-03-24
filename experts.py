import datetime
from ai_integration import generate_summary

conversation_log = []

def add_expert_message(expert, message):
    """Adds a message from an expert to the conversation log."""
    conversation_log.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "expert": expert,
        "message": message
    })

def generate_expert_response(expert, context, data_summary=None, model_choice="phi3"):
    """
    Generates an expert response using the local LLM.
    
    If a non-empty data_summary is provided, it will be included in the prompt.
    Otherwise, only the context is provided.
    The prompt instructs the model to produce a detailed and actionable analysis.
    """
    if data_summary and data_summary.strip():
        prompt = (
            f"You are {expert}, a seasoned expert in environmental sensor data analysis. "
            "Based on the following summarized data and context, provide a detailed, actionable analysis and forecast. "
            f"Data Summary: {data_summary}\n"
            f"Context: {context}\n\n"
            "Provide your expert analysis:"
        )
    else:
        prompt = (
            f"You are {expert}, a seasoned expert in environmental sensor data analysis. "
            "Based on the following context, provide a detailed, actionable, and specific forecast and analysis. "
            "Do not include generic placeholders or incomplete ranges. "
            f"Context: {context}\n\n"
            "Provide your expert analysis:"
        )
    response = generate_summary(prompt, model=model_choice)
    add_expert_message(expert, response)
    return response

def get_conversation_log():
    """Returns the conversation log."""
    return conversation_log
