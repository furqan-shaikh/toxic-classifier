import gradio as gr
import pandas as pd

from prediction_runner import run_prediction

def gradio_wrapper(model_type: str, raw_comments: str):
    comments = [c.strip() for c in raw_comments.split(",") if c.strip()]
    results = run_prediction(model_type, comments)

    rows = []
    for comment, preds in zip(comments, results):
        for pred in preds:
            rows.append({
                "Comment": comment,
                "Label": pred["label"],
                "Probability": pred["probability"],
                "Prediction": pred["prediction"]
            })

    return pd.DataFrame(rows)

def main():
    view = gr.Interface(
        fn=gradio_wrapper,
        inputs=[
                gr.Dropdown(["original_small(albert)"], label="Model Type", value="original_small(albert)"),
                gr.Textbox(lines=5, placeholder="Enter comments separated by comma", label="Comments")
        ],
        outputs=[gr.Dataframe(label="Predictions")],
        title="Toxic Comment Classifier")
    view.launch()

if __name__ =="__main__":
    main()

