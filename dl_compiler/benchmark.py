import json
from pathlib import Path
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
    model_name: str = "facebook/convnext-tiny-224",
    # model_name: str = "microsoft/resnet-50",
    image_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    batch_size: int = 16,
    epochs: int = 1000,
    warmup_epochs: int = 10,
    device="cuda",
    # test_item: str = "torch",
    # test_item: str = "torch_compile",
    test_item: str = "tvm",
    # test_item: str = "tvm_tune",
    result_json: str = "saved/{model_name_str}_{test_item}_{batch_size}_benchmark_result.json",
    log_file: str = "saved/{model_name_str}_{test_item}_{batch_size}_benchmark.log",
):
    """
    Reference:
    - https://huggingface.co/docs/transformers/v4.41.3/perf_torch_compile
    - https://tvm.apache.org/docs/how_to/compile_models/from_pytorch.html
    - https://tvm.apache.org/docs/how_to/tune_with_autotvm/tune_relay_cuda.html

    Available models for Image Classification
    - facebook/convnext-tiny-224
    - microsoft/resnet-50
    """
    kwargs_dic = locals()

    def format_path(x: str):
        p = Path(
            x.format(
                model_name_str=model_name.replace("/", "_"),
                test_item=test_item,
                batch_size=batch_size,
            )
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    result_json = format_path(result_json)
    log_file = format_path(log_file)
    logger.add(log_file)

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
        raise NotImplementedError
        result = test_tvm_tvm_tune(*common_args, is_tune=True)

    td_inference_arr = np.array(result["td_inference_lst"])
    logger.success(f"Inference: {td_inference_arr.mean()*1000:.2f} ms/batch")

    result["td_inference_mean"] = td_inference_arr.mean()
    result["td_inference_std"] = td_inference_arr.std()
    result["td_inference_median"] = np.median(td_inference_arr)

    result["kwargs_dic"] = kwargs_dic
    result_json.write_text(json.dumps(result, indent=4))


def test_torch_torch_compile(
    model_name: str,
    image_url: str,
    batch_size: int,
    epochs: int,
    warmup_epochs: int,
    device: str,
    is_compile: bool,
):

    result = {}
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
        result["td_compile"] = td_compile

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
    result["td_inference_lst"] = td_inference_lst

    return result


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

    result = {}
    image = Image.open(requests.get(image_url, stream=True).raw)
    images = [image] * batch_size
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForImageClassification.from_pretrained(
        model_name, torchscript=True
    )
    model.eval()
    logger.info(f"{model=}")

    processed_input = processor(images, return_tensors="pt")["pixel_values"]
    scripted_model = torch.jit.trace(model, processed_input).eval()

    input_name = "input0"
    shape_lst = [(input_name, processed_input.shape)]
    mod, params = relay.frontend.from_pytorch(scripted_model, shape_lst)

    dev = tvm.device(device)
    target = tvm.target.Target.from_device(dev)

    logger.debug("Compiling model")
    ti_compile0 = time.time()
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=target, params=params)
    td_compile = time.time() - ti_compile0
    logger.success(f"Build: {td_compile*1000:.2f} ms")
    result["td_compile"] = td_compile

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
    result["td_inference_lst"] = td_inference_lst

    return result


if __name__ == "__main__":
    fire.Fire(main)
