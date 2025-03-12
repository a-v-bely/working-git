import gradio as gr

# Custom CSS targeting MathJax3 elements
custom_css = """
mjx-container[jax="CHTML"][display="true"] {
    background: #f8f9fa !important;
    padding: 20px !important;
    border-radius: 8px !important;
    margin: 15px 0 !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    overflow-x: auto !important;
}

mjx-container[jax="CHTML"] {
    margin: 10px 0 !important;
}
"""

# MathJax configuration to ensure proper rendering
mathjax_script = """
<script>
MathJax = {
    tex: {
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']]
    }
};
</script>
"""

def format_equations(text):
    """Wrap block equations in proper MathJax delimiters"""
    return text.replace("\\[", "$$").replace("\\]", "$$")

def respond(message, chat_history):
    # Example response with formatted equations
    bot_message = "Here's an important equation: \\[E = mc^2\\]\n\nAnd another complex one:\n\\[\n\\int_0^\\infty x^2 dx = \\left.\\frac{x^3}{3}\\right|_0^\\infty\n\\]"
    formatted_message = format_equations(bot_message)
    chat_history.append((message, formatted_message))
    return "", chat_history

with gr.Blocks(css=custom_css, head=mathjax_script) as demo:
    gr.HTML("<script src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-mml-chtml.min.js'></script>")
    chatbot = gr.Chatbot(label="Math Chat", render_markdown=True)
    msg = gr.Textbox(label="Your Message")
    clear = gr.Button("Clear History")

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch()