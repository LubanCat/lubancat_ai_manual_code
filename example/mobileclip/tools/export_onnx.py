import torch
from PIL import Image
import mobileclip
import onnx

model, _, preprocess = mobileclip.create_model_and_transforms('mobileclip_s0', pretrained='path/models/mobileclip_s0.pt')
tokenizer = mobileclip.get_tokenizer('mobileclip_s0')

text_encoder = model.text_encoder
# text onnx
text_input = tokenizer("a photo of a cat", return_tensors="pt")
# print(text_input)
text_onnx_path = "text_encoder.onnx"
torch.onnx.export(text_encoder,
            (text_input),
            text_onnx_path,
            input_names=['text'],
            output_names=['text_features'],
            export_params=True,
            opset_version=13,
            verbose=False)

# onnx Checks
model_onnx = onnx.load(text_onnx_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model
    
# Save the image_encoder model
image_encoder = model.image_encoder
# image onnx
image = preprocess(Image.open("docs/fig_accuracy_latency.png").convert('RGB')).unsqueeze(0)
image_onnx_path = "image_encoder.onnx"
torch.onnx.export(image_encoder,
            (image),
            image_onnx_path,
            input_names=['image'],
            output_names=['image_features'],
            export_params=True,
            opset_version=13,
            verbose=False)

model_onnx = onnx.load(image_onnx_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model