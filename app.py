import gradio as gr
from utils.predictor import predict
from utils.explainer import lime_explanation
from utils.preprocess import clean_text
from utils.url_scraper import extract_from_url

def classify_text(input_text):
    clean = clean_text(input_text)
    label, confidence = predict(clean)
    explanation = lime_explanation(clean)
    return label, f"{confidence*100:.2f}%", explanation

def handle_url(url):
    text = extract_from_url(url)
    return text

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 FakeNewsX - Real vs Fake News Detector with Explainability")

    with gr.Tab("📝 Paste Text"):
        input_text = gr.Textbox(lines=8, label="Paste your news article text here")
        label = gr.Label(label="Prediction")
        confidence = gr.Label(label="Confidence")
        explanation = gr.HTML(label="LIME Explanation")

        submit_btn = gr.Button("Analyze News")
        submit_btn.click(classify_text, inputs=input_text, outputs=[label, confidence, explanation])

    with gr.Tab("🌐 Analyze URL"):
        url_input = gr.Textbox(label="News Article URL")
        url_output = gr.Textbox(label="Extracted Text", lines=8)
        scrape_btn = gr.Button("Extract from URL")
        scrape_btn.click(handle_url, inputs=url_input, outputs=url_output)

demo.launch()
