# Function to generate Python code using CodeLlama
def generate_code_with_codellama(client, description):
    """
    Generate Python code based on a natural language description using CodeLlama.

    Parameters:
    description (str): A plain-text description of the desired Python code.

    Returns:
    str: Generated Python code or an error message.
    """
    try:
        prompt = (
            f"You are a Python programming assistant. Based on the following description, "
            f"generate the Python code. Ensure the code is clear, well-commented, and includes necessary imports.\n\n"
            f"Description: {description}\n\n"
            f"Generated Python Code:"
        )

        # Call Together AI
        response = client.chat.completions.create(
            model="codellama/CodeLlama-34b-Instruct-hf",  # CodeLlama model
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract the generated code
        generated_code = response.choices[0].message.content.strip()
        return generated_code

    except Exception as e:
        return f"Error with CodeLlama: {e}"