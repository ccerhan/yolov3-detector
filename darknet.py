import torch
import torch.nn as nn
import numpy as np


class Darknet(nn.Module):
    """
    Darknet object detection module.
    """
    def __init__(self, config_path, input_size=None):
        """
        Initializes Darknet module with the configuration stored in 'config_path'.
        :param config_path: YOLOv3 configuration file path.
        :param input_size: Input image size (height, width). If None, image size in configs file will be used.
        """
        super(Darknet, self).__init__()
        self.module_defs = parse_model_config(config_path)
        if input_size is not None:
            self.module_defs[0]['height'] = input_size[0]
            self.module_defs[0]['width'] = input_size[1]
        self.module_list = create_modules(self.module_defs)

    def forward(self, x):
        """
        Forward-pass of the model.
        :param x: Input tensor (batch, channel, height, width).
        :return: Detection predictions (batch, num_predictions, num_class + 5)
        """
        output = []
        layer_outputs = []
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
            elif module_def['type'] == 'route':
                layer_i = [int(x) for x in module_def['layers'].split(',')]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif module_def['type'] == 'shortcut':
                layer_i = int(module_def['from'])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif module_def['type'] == 'yolo':
                x = module(x)
                output.append(x)
            layer_outputs.append(x)

        return torch.cat(output, 1)

    def load_weights(self, weights_path):
        """
        Parses and loads the weights stored in 'weights_path'.
        :param weights_path: YOLOv3 weights file path.
        """
        with open(weights_path, 'rb') as fp:
            header = np.fromfile(fp, dtype=np.int32, count=5)
            weights = np.fromfile(fp, dtype=np.float32)

        ptr = 0
        for i, (module_def, module) in enumerate(zip(self.module_defs, self.module_list)):
            if module_def['type'] == 'convolutional':
                conv_layer = module[0]
                if module_def['batch_normalize']:
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w


class YOLOLayer(nn.Module):
    """
    YOLO detection module.
    """
    def __init__(self, anchors, num_classes, input_size):
        """
        Initializes YOLO module.
        :param anchors: Anchor boxes.
        :param num_classes: Number of classes.
        :param input_size: Input image size (height, width).
        """
        super(YOLOLayer, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.input_size = input_size

        # Precomp grid offsets and anchor sizes
        self.grid_x = None
        self.grid_y = None
        self.anchor_w = None
        self.anchor_h = None

    def forward(self, x):
        na = len(self.anchors)
        nb, _, gh, gw = x.size()
        stride = self.input_size[0] / gh

        prediction = x.view(nb, na, self.bbox_attrs, gh, gw).permute(0, 1, 3, 4, 2).contiguous()

        cx = torch.sigmoid(prediction[..., 0])
        cy = torch.sigmoid(prediction[..., 1])
        bw = prediction[..., 2]
        bh = prediction[..., 3]
        conf_obj = torch.sigmoid(prediction[..., 4])
        conf_cls = torch.sigmoid(prediction[..., 5:])

        if self.grid_x is None:
            # Calculate offsets for each grid
            self.grid_x = torch.arange(gw, dtype=torch.float32).repeat(gh, 1).view([1, 1, gh, gw]).to(x.device)
            self.grid_y = torch.arange(gh, dtype=torch.float32).repeat(gw, 1).t().view([1, 1, gh, gw]).to(x.device)
            scaled_anchors = x.new([(aw / stride, ah / stride) for aw, ah in self.anchors])
            self.anchor_w = scaled_anchors[:, 0:1].view((1, na, 1, 1))
            self.anchor_h = scaled_anchors[:, 1:2].view((1, na, 1, 1))

        # Add offset and scale with anchors
        pred_boxes = x.new(prediction[..., :4].shape).float()
        pred_boxes[..., 0] = cx.data + self.grid_x
        pred_boxes[..., 1] = cy.data + self.grid_y
        pred_boxes[..., 2] = torch.exp(bw.data) * self.anchor_w
        pred_boxes[..., 3] = torch.exp(bh.data) * self.anchor_h

        output = torch.cat((pred_boxes.view(nb, -1, 4) * stride,
                            conf_obj.view(nb, -1, 1),
                            conf_cls.view(nb, -1, self.num_classes)), -1)

        return output


class UpsampleLayer(nn.Module):
    """
    Upsample module.
    Note: nn.Upsample gives deprecated warning message.
    """
    def __init__(self, scale_factor, mode='nearest'):
        super(UpsampleLayer, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return nn.functional.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class EmptyLayer(nn.Module):
    """
    Empty module which is a placeholder for 'route' and 'shortcut' layers.
    """
    def __init__(self):
        super(EmptyLayer, self).__init__()

    def forward(self, x):
        pass


def create_modules(module_defs):
    """
    Constructs module list of layer blocks from module configurations.
    :param module_defs: Module definitions parsed with 'parse_model_config'.
    :return: PyTorch module replacements of original Darknet layers.
    """
    hyperparams = module_defs.pop(0)
    output_filters = [int(hyperparams['channels'])]
    module_list = nn.ModuleList()
    for i, module_def in enumerate(module_defs):
        modules = nn.Sequential()

        if module_def['type'] == 'convolutional':
            bn = int(module_def['batch_normalize'])
            filters = int(module_def['filters'])
            kernel_size = int(module_def['size'])
            pad = (kernel_size - 1) // 2 if int(module_def['pad']) else 0
            modules.add_module(
                'conv_%d' % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(module_def['stride']),
                    padding=pad,
                    bias=not bn,
                ),
            )
            if bn:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if module_def['activation'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))

        elif module_def['type'] == 'maxpool':
            kernel_size = int(module_def['size'])
            stride = int(module_def['stride'])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module('_debug_padding_%d' % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(module_def['size']),
                stride=int(module_def['stride']),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module('maxpool_%d' % i, maxpool)

        elif module_def['type'] == 'upsample':
            upsample = UpsampleLayer(scale_factor=int(module_def['stride']), mode='nearest')
            modules.add_module('upsample_%d' % i, upsample)

        elif module_def['type'] == 'route':
            layers = [int(x) for x in module_def['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            modules.add_module('route_%d' % i, EmptyLayer())

        elif module_def['type'] == 'shortcut':
            filters = output_filters[int(module_def['from'])]
            modules.add_module('shortcut_%d' % i, EmptyLayer())

        elif module_def['type'] == 'yolo':
            anchor_idxs = [int(x) for x in module_def['mask'].split(',')]
            # Extract anchors
            anchors = [int(x) for x in module_def['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(module_def['classes'])
            width = int(hyperparams['width'])
            height = int(hyperparams['height'])
            # Define detection layer
            yolo_layer = YOLOLayer(anchors, num_classes, (height, width))
            modules.add_module('yolo_%d' % i, yolo_layer)

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    return module_list


def parse_model_config(path):
    """
    Parses the configuration file.
    :param path: YOLOv3 configuration file path.
    :return: Module definitions as an ordered list.
    """
    with open(path, 'r') as fp:
        lines = fp.read().split('\n')
        lines = [x.rstrip().lstrip() for x in lines if x and not x.startswith('#')]

    module_defs = []
    for line in lines:
        if line.startswith('['):
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0
        else:
            key, value = line.split('=')
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs
