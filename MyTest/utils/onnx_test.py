# --- convert to onnx--- #
import torch.onnx
import importlib
import timm

from util import get_config
from MyTest.models.network import get_network


#  Function to Convert to ONNX
def Convert_ONNX(model, dummy_input, save_path):
    # set the model to inference mode
    model.eval()

    # Export the model
    torch.onnx.export(model,  # model being run
                      dummy_input,  # model input (or a tuple for multiple inputs)
                      save_path,  # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['modelInput'],  # the model's input names
                      output_names=['modelOutput'],  # the model's output names
                      dynamic_axes={'modelInput': {0: 'batch_size'},  # variable length axes
                                    'modelOutput': {0: 'batch_size'}})
    print(" ")
    print('Model has been converted to ONNX')


def exec_convert_onnx():
    config = importlib.import_module("MyTest.configs.config")
    importlib.reload(config)
    cfg = config.config

    # checkpoint_path = '../checkpoints/mobile_net/Sep22/net_39.pth'
    checkpoint_path = '../checkpoints/Sep20/net_39.pth'

    # onnx_save_path = '../checkpoints/mobile_net/Sep22/mobilenet.onnx'
    onnx_save_path = '../checkpoints/Sep20/resnet.onnx'

    # net = timm.create_model('mobilenetv2_100', num_classes=1500)
    net = get_network(cfg)

    net.load_state_dict(torch.load(checkpoint_path))

    dummy_input = torch.randn(1, 3, 256, 256, requires_grad=True)

    Convert_ONNX(model=net, dummy_input=dummy_input, save_path=onnx_save_path)


def infer_onnx():
    import onnx
    onnx_model = onnx.load('../checkpoints/mobile_net/Sep22/mobilenet.onnx')
    onnx.checker.check_model(onnx_model)
    print('checked')


if __name__ == '__main__':
    # convert to onnx
    # exec_convert_onnx()

    # onnx inference
    infer_onnx()




