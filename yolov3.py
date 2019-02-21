import torch
import torch.nn as nn
from darknet import Darknet


class Detector(nn.Module):
    def __init__(self, config_path, weights_path, input_size=None, conf_thresh=0.5, nms_thresh=0.4):
        super(Detector, self).__init__()
        self.input_size = input_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        # Initialize Darknet for detection
        self.model = Darknet(config_path, input_size=input_size)
        self.model.load_weights(weights_path)
        self.model.eval()

    def device(self):
        return next(self.model.parameters()).device

    def forward(self, frame, swapRB=False):
        x = image_to_tensor(frame, swapRB)
        _, _, fh, fw = x.size()

        device = self.device()
        x = x.to(device)
        x = letterbox_resize(x, self.input_size, constant_value=127.5)
        x = x / 255.0

        with torch.no_grad():
            y = self.model.forward(x)

        output = []
        for i, prediction in enumerate(y):  # Enumerate on batch
            detection = non_max_suppression(prediction.cpu(), self.conf_thresh, self.nms_thresh)
            if detection is not None:
                detection = bbox_fit(detection, (fh, fw), self.input_size).to(device)
            output.append(detection)

        return output

    def update(self, conf_thresh=None, nms_thresh=None, weights_path=None):
        if conf_thresh is not None:
            self.conf_thresh = conf_thresh

        if nms_thresh is not None:
            self.nms_thresh = nms_thresh

        if weights_path is not None:
            device = self.device()
            self.model.cpu().load_weights(weights_path)
            self.model.to(device)


def image_to_tensor(input, swapRB=False):
    """
    :param input: Image which size is (height, width, channel) or (batch, height, width, channel)
    :param swapRB: If true, red and blue channels will be swapped.
    :return: Torch tensor which size is (batch, channel, height, width)
    """
    if swapRB:
        pass  # input = input[:, :, ::-1].copy() # TODO: Not tested...

    if len(input.shape) == 3:
        return torch.from_numpy(input.transpose(2, 0, 1)).float().unsqueeze(0)
    elif len(input.shape) == 4:
        return torch.from_numpy(input.transpose(3, 1, 2)).float()


def letterbox_resize(input, size, constant_value=0, resize_mode='nearest'):
    ih, iw = input.size(-2), input.size(-1)
    sh, sw = size

    pad_h = int(max((iw * sh / sw - ih) // 2, 0))
    pad_w = int(max((ih * sw / sh - iw) // 2, 0))
    pad = (pad_w, pad_w, pad_h, pad_h)

    output = nn.functional.pad(input, pad=pad, mode='constant', value=constant_value)
    output = nn.functional.interpolate(output, size=(sh, sw), mode=resize_mode,
                                       align_corners=(False if resize_mode == 'bilinear' else None))

    return output


def bbox_fit(detection, frame_size, input_size):
    """
        Accepts detections with shape: (x1, y1, x2, y2, ... )
    """
    fh, fw = frame_size
    ih, iw = input_size

    fdim = max(fh, fw)
    pad_x = max((fh * iw / ih - fw) / 2, 0) * iw / fdim
    pad_y = max((fw * ih / iw - fh) / 2, 0) * ih / fdim
    scale_x = fw / (iw - 2 * pad_x)
    scale_y = fh / (ih - 2 * pad_y)

    detection[:, 0] = torch.clamp((detection[:, 0] - pad_x) * scale_x, min=0, max=fw)
    detection[:, 1] = torch.clamp((detection[:, 1] - pad_y) * scale_y, min=0, max=fh)
    detection[:, 2] = torch.clamp((detection[:, 2] - pad_x) * scale_x, min=0, max=fw)
    detection[:, 3] = torch.clamp((detection[:, 3] - pad_y) * scale_y, min=0, max=fh)

    return detection


def bbox_iou(box1, box2):
    """
    Accepts boxes with shape: (x1, y1, x2, y2, ... )
    Returns the IoU of two bounding boxes
    """
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    w = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0)
    h = torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
    inter_area = w * h

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


def non_max_suppression(prediction, conf_thresh=0.5, nms_thresh=0.4):
    """
        Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        Returns detections with shape: (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """
    # Apply lower bound confidence threshold
    indices = (prediction[:, 4] >= conf_thresh).nonzero().view(-1)
    prediction = torch.index_select(prediction, 0, indices)
    if not prediction.size(0):
        return None

    # (cx, cy, bw, bh) -> (x1, y1, x2, y2)
    box_corner = prediction.new(prediction.shape)
    box_corner[:, 0] = prediction[:, 0] - prediction[:, 2] / 2
    box_corner[:, 1] = prediction[:, 1] - prediction[:, 3] / 2
    box_corner[:, 2] = prediction[:, 0] + prediction[:, 2] / 2
    box_corner[:, 3] = prediction[:, 1] + prediction[:, 3] / 2
    prediction[:, :4] = box_corner[:, :4]

    # Get score and class with highest confidence
    class_conf, class_pred = torch.max(prediction[:, 5: prediction.size(-1)], 1, keepdim=True)

    # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    detections = torch.cat((prediction[:, :5], class_conf.float(), class_pred.float()), 1)

    output = []
    for c in detections[:, -1].unique():
        # Get the detections with the particular class
        detections_class = detections[detections[:, -1] == c]

        # Sort the detections by maximum objectness confidence
        _, conf_sort_index = torch.sort(detections_class[:, 4], descending=True)
        detections_class = detections_class[conf_sort_index]

        # Perform non-maximum suppression
        max_detections = []
        while detections_class.size(0):
            # Get detection with highest confidence and save as max detection
            max_detections.append(detections_class[0].unsqueeze(0))
            # Stop if we're at the last detection
            if len(detections_class) == 1:
                break

            ious = bbox_iou(max_detections[-1], detections_class[1:])

            # Apply NMS threshold
            detections_class = detections_class[1:][ious < nms_thresh]

        output.append(torch.cat(max_detections))

    return torch.cat(output)
