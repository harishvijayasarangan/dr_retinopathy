import torch
import gradio as gr
from PIL import Image
from torchvision import transforms as T
CHECKPOINT_PATH = r"C:\Users\STIC-11\Desktop\sk1\densenet_retinopathy.pth"
model = DRModel.load_from_checkpoint(CHECKPOINT_PATH, map_location="cpu")
model.eval()
labels = {
    0: "No DR",
    1: "Mild",
}
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
def predict(image):
    image = image.convert("RGB")
    input_img = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(input_img)[0], dim=0)
        confidences = {labels[i]: float(prediction[i]) for i in labels}
    return confidences
dr_app = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(),
    title="Diabetic Retinopathy Detection",
    description="Upload a retinal image to get a prediction of diabetic retinopathy severity."
)

if __name__ == "__main__":
    dr_app.launch()