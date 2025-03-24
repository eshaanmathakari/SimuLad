import subprocess

def generate_summary(simulation_text, model="phi3"):
    """
    Uses the local AI model (via Ollama) to generate a natural language summary.
    simulation_text: text description of simulation results.
    Returns the AI-generated summary.
    """
    prompt = f"Summarize the following simulation results: {simulation_text}"
    try:
        result = subprocess.run(
            ["ollama", "run", model, prompt],
            capture_output=True,
            text=True,
            check=True
        )
        summary = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        summary = f"Error generating summary: {e}"
    return summary
