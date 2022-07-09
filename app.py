"""
Source: https://towardsdatascience.com/how-to-deploy-a-machine-learning-ui-on-heroku-in-5-steps-b8cd3c9208e6
"""

from fastai.text.all import load_learner
import gradio as gr

def predict(inp):
    model = load_learner("./export.pkl")
    prediction = model.predict(inp, 32, temperature=1)
    return prediction

gr.Interface(
    fn=predict,
    inputs=gr.inputs.Textbox(),
    outputs=gr.outputs.Textbox(),
    examples=["The film was", "What a waste", "I have to admit"]
).launch(inbrowser=True)