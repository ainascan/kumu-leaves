import time
import os

import torch as t
import torch.nn as nn
import torchvision as tv
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import MaskRCNN
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights   
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import nms as non_max_suppression
from torch.utils.data import DataLoader
from tqdm import tqdm
import mlflow


def to_cpu(images, targets):
    """
    Have to move things to the CPU when we are done.
    """
    images = images.to('cpu')
    for target in targets:
        for key in target:
            target[key] = target[key].to('cpu')


def to_gpu(images, boxes, masks, labels):
    """
    Need to move things to the GPU and convert them to the right data types.
    """
    images = t.stack(images).float().cuda()
    targets = [
        {
            'masks': t.stack(m).type(t.uint8).cuda(),
            'labels': t.tensor(l).cuda(),
            'boxes': t.stack(b).squeeze(1).cuda(),
        } for b, m, l in zip(boxes, masks, labels)
    ]
    return images, targets


def get_backbone(name):
    """
    We need to set the out_channels attribute of the backbone to the number of channels in the final feature map.
    If you print the model, you will see the last layer of the backbone and know how many channels it has.
    """
    if name == 'mobilenet_v2':
        backbone = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2).features
        backbone.out_channels = 1280
        return backbone
    
    if name == 'efficientnet_b0':
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1).features
        backbone.out_channels = 1280
        return backbone

    if name == 'efficientnet_v2_s':
        backbone = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1).features
        backbone.out_channels = 1280
        return backbone

    if name == 'efficientnet_b3':
        backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1).features
        backbone.out_channels = 1536
        return backbone
    
    if name == 'mobilenet_v3_small':
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1).features
        backbone.out_channels = 576
        return backbone
    
    if name == 'resnet50':
        backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.IMAGENET1K_V1, trainable_layers=2)
        return backbone
    
    if name == 'resnet18':
        backbone = resnet_fpn_backbone('resnet18', weights=ResNet18_Weights.IMAGENET1K_V1, trainable_layers=2)
        return backbone
    
    
    
def filter_with_confidence(pred, mask_height, mask_width, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Filter out predictions with confidence below the threshold and apply non-maximum suppression.
    """
    nms_indicies = non_max_suppression(pred['boxes'], pred['scores'], iou_threshold)
    pred = {key: pred[key][nms_indicies] for key in pred}
    
    # filter out predictions with confidence below the threshold.
    remaining = pred['scores'] > confidence_threshold
    
    # we default to empty tensors if no predictions are left.
    conf_preds = {
        'boxes': t.empty((0, 4)),
        'masks': t.empty((0, mask_height, mask_width)).type(t.uint8),
        'labels': t.empty(0),
        'scores': t.empty(0),
    }
    if remaining.sum():
        conf_preds = {
            'boxes': pred['boxes'][remaining],
            'masks': (pred['masks'][remaining] > 0.5).type(t.uint8),
            'labels': pred['labels'][remaining],
            'scores': pred['scores'][remaining],
        }
        
    return conf_preds


class LaauModel(nn.Module):
    
    def __init__(self,
        model_path: str,
        save_path: str,
        backbone='mobilenet_v3_small',
        freeze_at=2,
        anchor_sizes=(32, 64, 128, 256),
        anchor_ratios=(0.5, 1.0, 2.0),
        max_size=1920,
        min_size=1080,
        resume=False
    ):
        super(LaauModel, self).__init__()
        
        self.model_path = model_path
        self.save_path = save_path
        self.freeze_at = freeze_at
        
        anchor_generator = AnchorGenerator(
            sizes=(tuple(anchor_sizes),),
            aspect_ratios=(tuple(anchor_ratios),)
        )
        
        backbone = get_backbone(backbone)

        self.model = MaskRCNN(
            backbone=backbone,
            max_size=max_size,
            min_size=min_size,
            num_classes=2,
            rpn_anchor_generator=anchor_generator,
        )
        
        for param in self.model.parameters():
            param.requires_grad = False
            param.grad = None

        # unfreeze the last freeze_at layers of the backbone.
        children = list(self.model.backbone.named_children())
        for _, child in children[-freeze_at:]:
            for param in child.parameters():
                param.requires_grad = True

        # move to gpu, compile the model, and warm it up.
        self.model = self.model.float().cuda()
        self.model.eval()
        input = t.rand(3, 3, 224, 224).cuda()
        with t.cuda.amp.autocast():
            self.model(input)

        os.makedirs(self.save_path, exist_ok=True)

        if resume:
            self.model.load_state_dict(t.load(self.model_path))


    def get_optimizer(self, learning_rate):
        learnable_params = []
        for param in self.model.backbone[-self.freeze_at:].parameters():
            param.requires_grad = True
            learnable_params.append(param)
        
        return t.optim.Adam(learnable_params, lr=learning_rate)


    def forward(self, x):
        return self.model(x['image'], x['targets'])


    def save(self, name):
        path = os.path.join(self.save_path, name)
        t.save(self.model.state_dict(), path)


    def validate(
        self,
        title: str,
        dataset: DataLoader,
        epoch: int,
        iou_threshold=0.5,
        confidence_threshold=0.6
    ):
        metrics = {
            'box_ap': MeanAveragePrecision(iou_type='bbox'),
            'mask_ap': MeanAveragePrecision(iou_type='segm'),
        }

        self.model.eval()
        
        progress = tqdm(total=len(dataset.dataset), desc=f'{title}', position=0, leave=True)
        
        with t.no_grad():

            for frames, boxes, masks, labels in dataset:
                frames, targets = to_gpu(frames, boxes, masks, labels)

                height, width = frames.shape[-2:]

                with t.cuda.amp.autocast():
                    outputs = self.model(frames)
                
                to_cpu(frames, targets)
                outputs = [{k: v.cpu() for k, v in output.items()} for output in outputs]
                t.cuda.empty_cache()
                
                for index in range(len(targets)):
                    pred = outputs[index]
                    target = targets[index]
                    pred['masks'] = (pred['masks'] > 0.5).type(t.uint8).squeeze()
                    
                    conf_preds = filter_with_confidence(pred, height, width, confidence_threshold, iou_threshold)
                    
                    metrics['mask_ap'].update([conf_preds], [target])
                    metrics['box_ap'].update([conf_preds], [target])

                progress.update(len(frames))
                progress.refresh()

        mask_ap = metrics['mask_ap'].compute()
        box_ap = metrics['box_ap'].compute()
        
        val_metrics = {
            'mask_ap': f"{mask_ap.get('map', 0):.4f}",
            'mask_ap50': f"{mask_ap.get('map_50', 0):.4f}",
            'mask_ap75': f"{mask_ap.get('map_75', 0):.4f}",
            'box_ap': f"{box_ap.get('map', 0):.4f}",
            'box_ap50': f"{box_ap.get('map_50', 0):.4f}",
            'box_ap75': f"{box_ap.get('map_75', 0):.4f}",
        }
        
        for metric in val_metrics:
            mlflow.log_metric(key=f'{title}_{metric}'.lower(), value=val_metrics[metric], step=epoch)
        
        progress.set_postfix(**val_metrics)
        progress.close()


    def train(
        self,
        train_dataset: DataLoader,
        val_dataset: DataLoader,
        test_dataset: DataLoader,
        learning_rate=0.001,
        save_epochs=10,
        epochs=1000,
        val_epochs=10,
        iou_threshold=0.5,
        confidence_threshold=0.6
    ):
    
        with t.autograd.set_detect_anomaly(False):
            optimizer = self.get_optimizer(learning_rate)
            scheduler = t.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
            scaler = t.cuda.amp.GradScaler()

            for epoch in range(epochs):
                
                self.model.train()

                progress = tqdm(total=len(train_dataset.dataset), desc='Training', position=0, leave=True)
                epoch_start = time.time()
                running_loss = 0.0
                total_batches = 0
                
                for frames, bboxes, masks, labels in train_dataset:
                    frames, targets = to_gpu(frames, bboxes, masks, labels)

                    optimizer.zero_grad()
                    with t.cuda.amp.autocast():
                        loss = self.model(frames, targets)
                        loss = sum(loss.values())
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    running_loss += loss
                    total_batches += 1

                    to_cpu(frames, targets)
                    t.cuda.empty_cache()
                    
                    progress.set_postfix(loss=f'{loss:.6f}')
                    progress.update(len(frames))
                    progress.refresh()

                epoch_loss = running_loss / total_batches
                epoch_time = time.time() - epoch_start
                
                mlflow.log_metric(key='epoch_loss', value=epoch_loss, step=epoch+1)
                mlflow.log_metric(key='epoch_time', value=epoch_time, step=epoch+1)

                progress.set_postfix(epoch=f'{epoch+1}/{epochs}', loss=f'{epoch_loss:.6f}', time=f'{epoch_time:.2f}s')
                progress.close()
                scheduler.step()

                t.cuda.empty_cache()
                
                if epoch > 0 and epoch % save_epochs == 0:
                    self.save(f'model_{epoch}.pth')
                
                if epoch > 0 and epoch % val_epochs == 0:
                    self.validate('Validation', val_dataset, epoch+1, iou_threshold, confidence_threshold)

            t.cuda.empty_cache()

            self.save('model_final.pth')
            self.validate('Test', test_dataset, epochs, iou_threshold, confidence_threshold)

