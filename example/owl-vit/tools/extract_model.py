import onnx

# owlvit-image.onnx
input_path = "owlvit_onnx/model.onnx"
output_path = "owlvit_onnx/owlvit-image.onnx"
input_names = ["pixel_values"]
output_names = ["image_embeds","pred_boxes"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)

# owlvit-text.onnx
output_path = "owlvit_onnx/owlvit-text.onnx"
input_names = ["/layer_norm/LayerNormalization_output_0", "input_ids", "attention_mask"]
output_names = ["logits"]

onnx.utils.extract_model(input_path, output_path, input_names, output_names)