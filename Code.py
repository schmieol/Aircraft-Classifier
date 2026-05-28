import gradio as gr
from transformers import pipeline


pipeline = pipeline(task="image-classification", model="pranavanbupathy/test-cifar-10")

def predict(image):
    predictions = pipeline(image)
    result = predictions[0]
    return result["label"]

gr.Interface(
    fn=predict,
    inputs=gr.Image(label="Upload only commercial aircraft image here", type="filepath"),
    outputs=gr.Label(label="Predicted Aircraft Type"),
    examples=[
        "./Bombardier-CRJ-700.jpg",
        "./atr72.jpeg",
        "./c208.jpg",
        "./fokker70.jpg",
        "b737.jpg"
    ],
    title="Find which Aircraft it is ✈️",
    description="Upload an image of a commercial aircraft and the model will predict its type."
).launch(inline=False)
