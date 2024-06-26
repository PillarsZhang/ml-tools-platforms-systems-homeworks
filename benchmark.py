import fire
import torch
from PIL import Image
import requests
import numpy as np
from transformers import AutoImageProcessor, AutoModelForImageClassification
from loguru import logger
import time

torch.set_float32_matmul_precision("high")
AVAILABLE_TEST_ITEMS = ["torch", "torch_compile", "tvm", "tvm_tune"]


def main(
    # model_name: str = "google/vit-base-patch16-224",
    model_name: str = "microsoft/resnet-50",
    image_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    batch_size: int = 64,
    epochs: int = 1000,
    warmup_epochs: int = 100,
    device="cuda",
    # test_item: str = "torch",
    # test_item: str = "torch_compile",
    test_item: str = "tvm",
    # test_item: str = "tvm_tune",
):
    """
    Reference:
    - https://huggingface.co/docs/transformers/v4.41.3/perf_torch_compile
    - https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html
    - https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_cuda.html

    Available models for Image Classification
    - google/vit-base-patch16-224
    - microsoft/beit-base-patch16-224-pt22k-ft22k
    - facebook/convnext-large-224
    - microsoft/resnet-50
    """
    assert (
        test_item in AVAILABLE_TEST_ITEMS
    ), f"{test_item=} not in {AVAILABLE_TEST_ITEMS}"

    common_args = (model_name, image_url, batch_size, epochs, warmup_epochs, device)
    if test_item == "torch":
        result = test_torch_torch_compile(*common_args, is_compile=False)
    elif test_item == "torch_compile":
        result = test_torch_torch_compile(*common_args, is_compile=True)
    elif test_item == "tvm":
        result = test_tvm_tvm_tune(*common_args, is_tune=False)
    elif test_item == "tvm_tune":
        result = test_tvm_tvm_tune(*common_args, is_tune=True)

    td_inference_arr = np.array(result["td_inference_lst"])
    logger.success(f"Inference: {td_inference_arr.mean()*1000:.2f} ms/batch")


def test_torch_torch_compile(
    model_name: str,
    image_url: str,
    batch_size: int,
    epochs: int,
    warmup_epochs: int,
    device: str,
    is_compile: bool,
):

    image = Image.open(requests.get(image_url, stream=True).raw)
    images = [image] * batch_size
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(model_name).to(device)
    model.eval()
    logger.info(f"{model=}")
    input = processor(images, return_tensors="pt").to(device)["pixel_values"]
    logger.info(f"{input.shape=}")

    if is_compile:
        logger.debug("Compiling model")
        ti_compile0 = time.time()
        model = torch.compile(model)
        td_compile = time.time() - ti_compile0
        logger.success(f"Compile: {td_compile*1000:.2f} ms")

    logger.debug(f"Warmup: {warmup_epochs} epochs")
    for _ in range(warmup_epochs):
        with torch.no_grad():
            _ = model(input)
            torch.cuda.synchronize()
    logger.debug("Warmup done")

    logger.debug(f"Timing: {epochs} epochs")
    td_inference_lst = []
    for _ in range(epochs):
        with torch.no_grad():
            ti_inference0 = time.time()
            _ = model(input)
            torch.cuda.synchronize()
            td_inference_lst.append(time.time() - ti_inference0)
    logger.debug("Timing done")

    return {"td_inference_lst": td_inference_lst}


def test_tvm_tvm_tune(
    model_name: str,
    image_url: str,
    batch_size: int,
    epochs: int,
    warmup_epochs: int,
    device: str,
    is_tune: bool,
):

    import tvm.relay as relay
    import tvm
    from tvm.contrib import graph_executor

    image = Image.open(requests.get(image_url, stream=True).raw)
    images = [image] * batch_size
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, torchscript=True
    ).to(device)
    model.eval()
    logger.info(f"{model=}")

    processed_input = processor(images, return_tensors="pt").to(device)["pixel_values"]
    scripted_model = torch.jit.trace(model, processed_input).eval()

    input_name = "input0"
    shape_lst = [(input_name, processed_input.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_lst)

    dev = tvm.device(device)
    target = tvm.target.Target.from_device(dev)

    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)

    m = graph_executor.GraphModule(lib["default"](dev))
    m.set_input(input_name, tvm.nd.array(processed_input.cpu(), dev))

    logger.debug(f"Warmup: {warmup_epochs} epochs")
    for _ in range(warmup_epochs):
        m.run()
    logger.debug("Warmup done")

    logger.debug(f"Timing: {epochs} epochs")
    td_inference_lst = []
    for _ in range(epochs):
        ti_inference0 = time.time()
        m.run()
        td_inference_lst.append(time.time() - ti_inference0)
    logger.debug("Timing done")

    return {"td_inference_lst": td_inference_lst}


if __name__ == "__main__":
    fire.Fire(main)
