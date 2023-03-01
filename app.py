import numpy as np
import cv2
import onnxruntime
import gradio as gr


def pre_process(img: np.array) -> np.array:
    # H, W, C -> C, H, W
    img = np.transpose(img[:, :, 0:3], (2, 0, 1))
    # C, H, W -> 1, C, H, W
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img


def post_process(img: np.array) -> np.array:
    # 1, C, H, W -> C, H, W
    img = np.squeeze(img)
    # C, H, W -> H, W, C
    img = np.transpose(img, (1, 2, 0))[:, :, ::-1].astype(np.uint8)
    return img


def inference(model_path: str, img_array: np.array) -> np.array:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    ort_session = onnxruntime.InferenceSession(model_path, options)
    ort_inputs = {ort_session.get_inputs()[0].name: img_array}
    ort_outs = ort_session.run(None, ort_inputs)

    return ort_outs[0]


def convert_pil_to_cv2(image):
    # pil_image = image.convert("RGB")
    open_cv_image = np.array(image)
    # RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def upscale(image, model):
    model_path = f"models/{model}.ort"
    img = convert_pil_to_cv2(image)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if img.shape[2] == 4:
        alpha = img[:, :, 3]  # GRAY
        alpha = cv2.cvtColor(alpha, cv2.COLOR_GRAY2BGR)  # BGR
        alpha_output = post_process(inference(model_path, pre_process(alpha)))  # BGR
        alpha_output = cv2.cvtColor(alpha_output, cv2.COLOR_BGR2GRAY)  # GRAY

        img = img[:, :, 0:3]  # BGR
        image_output = post_process(inference(model_path, pre_process(img)))  # BGR
        image_output = cv2.cvtColor(image_output, cv2.COLOR_BGR2BGRA)  # BGRA
        image_output[:, :, 3] = alpha_output

    elif img.shape[2] == 3:
        image_output = post_process(inference(model_path, pre_process(img)))  # BGR

    return image_output


css = ".output-image, .input-image, .image-preview {height: 480px !important} "
model_choices = ["modelx2", "modelx2 25 JXL", "modelx4", "minecraft_modelx4"]

gr.Interface(
    fn=upscale,
    inputs=[
        gr.inputs.Image(type="pil", label="Input Image"),
        gr.inputs.Radio(
            model_choices,
            type="value",
            default="modelx2",
            label="Choose Upscaler",
            optional=False,
        ),
    ],
    outputs="image",
    title="",
    description="",
    allow_flagging="never",
    css=css,
).launch()
