import random
import os.path
import urllib.request
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import yolov3 as yolo


def download_weights(dst):
    def progress(count, block, total):
        percent = int(count * block * 100 / total)
        print('Downloading yolov3.weights into data/yolov3.weights... [{}%]'.format(percent))
    src = 'https://pjreddie.com/media/files/yolov3.weights'
    urllib.request.urlretrieve(src, dst, reporthook=progress)


def load_labels(path):
    with open(path) as file:
        labels = [line.rstrip('\n') for line in file]
        colors = ['#{:06x}'.format(random.randint(0, 0xFFFFFF)) for i in range(len(labels))]
        return labels, colors


def main():
    weights_path = 'data/yolov3.weights'
    config_path = 'data/yolov3.cfg'
    labels_path = 'data/coco.names'

    # Download yolov3.weights if it does not exist
    if not os.path.exists(weights_path):
        download_weights(weights_path)

    # Load the class labels and randomly generated colors
    labels, colors = load_labels(labels_path)

    # Load the sample image as numpy array (RGB)
    image = plt.imread('samples/dog.jpg')

    # Create YOLO detector
    model = yolo.Detector(config_path=config_path,
                          weights_path=weights_path,
                          input_size=(544, 608),
                          conf_thresh=0.5,
                          nms_thresh=0.4)

    if torch.cuda.is_available():
        model.cuda()

    # Perform detection for a single image
    detections = model(image)

    # Draw detected class labels and relevant bounding boxes
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    if len(detections) > 0:
        for i, (x1, y1, x2, y2, obj_conf, cls_conf, cls_pred) in enumerate(detections[0]):
            x = round(x1.item())
            y = round(y1.item())
            w = round(x2.item() - x1.item())
            h = round(y2.item() - y1.item())

            label = labels[int(cls_pred)]
            color = colors[int(cls_pred)]

            ax.add_patch(patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none'))
            plt.text(x, y, s=label, color='white', verticalalignment='top', bbox={'color': color, 'pad': 0})

            print(i, ':', label, 'x:', x, 'y:', y, 'w:', w, 'h:', h)

    plt.axis('off')
    plt.gca().xaxis.set_major_locator(ticker.NullLocator())
    plt.gca().yaxis.set_major_locator(ticker.NullLocator())

    plt.show()
    # plt.savefig('samples/dogs_.png', bbox_inches='tight', pad_inches=0.0)

    plt.close()


if __name__ == '__main__':
    main()

