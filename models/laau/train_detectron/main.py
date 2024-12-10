import os
import time
import copy
import logging
import torch
import detectron2
import numpy as np
import cv2
import argparse
import yaml
import mlflow

from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.evaluation import COCOEvaluator
from detectron2.config import get_cfg, CfgNode
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data import detection_utils, build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.solver.build import get_default_optimizer_params, maybe_add_gradient_clipping

# hack cuz detectron2 uses np.bool which is deprecated
np.bool = np.bool_

print("torch: ", torch.__version__)
print("detectron2:", detectron2.__version__)
print("numpy:", np.__version__)
print("cv2:", cv2.__version__)


def custom_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)

    image = cv2.imread(dataset_dict["file_name"])
    
    modified_annotations = []
    for annotation in dataset_dict["annotations"]:
        contour = np.array(annotation['segmentation']).reshape(-1, 2).astype(np.int32)

        annotation['segmentation'] = [contour.flatten().tolist()]
        annotation['bbox'] = cv2.boundingRect(contour)
        modified_annotations.append(annotation)
    
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    instances = detection_utils.annotations_to_instances(modified_annotations, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)

    return dataset_dict


class CustomTrainer(DefaultTrainer):
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)
    
    # @classmethod
    # def build_optimizer(cls, cfg, model):
    #     """
    #     Build an optimizer from config.
    #     """
    #     params = get_default_optimizer_params(
    #         model,
    #         base_lr=cfg.SOLVER.BASE_LR,
    #         weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
    #         bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
    #         weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
    #     )
    #     return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
    #         params, 
    #         lr=cfg.SOLVER.BASE_LR,
    #         weight_decay=cfg.SOLVER.WEIGHT_DECAY
    #     )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(os.path.join(cfg.OUTPUT_DIR, "validation_evaluation"), exist_ok=True)

        return COCOEvaluator(dataset_name, distributed=False, output_dir=os.path.join(cfg.OUTPUT_DIR, "validation_evaluation"))


class MLflowHook(HookBase):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                mlflow.log_metric(key=k, value=v[0], step=v[1])


def condence_tags(cfg, key_prefix=''):
    tags = []
    for key, value in cfg.items():
        if type(value) == CfgNode:
            tags.extend(condence_tags(value, f'{key_prefix}.{key}' if key_prefix else key))
        else:
            tags.append([f'{key_prefix}.{key}' if key_prefix else key, value])
    return tags


def register_datasets(configuration):
    coco_dataset_dir = configuration['coco_dataset_dir']
    register_coco_instances("train", {}, os.path.join(coco_dataset_dir, 'train.json'), coco_dataset_dir)
    register_coco_instances("test", {}, os.path.join(coco_dataset_dir, 'test.json'), coco_dataset_dir)
    register_coco_instances("validation", {}, os.path.join(coco_dataset_dir, 'validation.json'), coco_dataset_dir)


def train_model(run_name, configuration):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(configuration['detectron']['model_arch']))

    # https://github.com/facebookresearch/detectron2/blob/main/detectron2/config/defaults.py
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = configuration['detectron']['score_thresh_test']
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = configuration['detectron']['nms_thresh_test']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = configuration['detectron']['batch_size_per_image']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = configuration['detectron']['number_of_classes']
    cfg.MODEL.BACKBONE.FREEZE_AT = configuration['detectron']['freeze_at']
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = configuration['detectron']['anchor_sizes']
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = configuration['detectron']['anchor_ratios']
    cfg.MODEL.WEIGHTS = configuration['detectron']['model_weights']

    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("validation",)

    cfg.INPUT.CROP.ENABLED = configuration['detectron']['crop_enabled']
    cfg.INPUT.RANDOM_FLIP = configuration['detectron']['random_flip']
    cfg.INPUT.MAX_SIZE_TRAIN = configuration['detectron']['max_size_train']
    cfg.INPUT.MIN_SIZE_TRAIN = configuration['detectron']['min_size_train']
    cfg.INPUT.MAX_SIZE_TEST = configuration['detectron']['max_size_test']
    cfg.INPUT.MIN_SIZE_TEST = configuration['detectron']['min_size_test']

    cfg.SOLVER.MAX_ITER = configuration['detectron']['epochs']
    cfg.SOLVER.CHECKPOINT_PERIOD = configuration['detectron']['checkpoint_period']
    cfg.SOLVER.IMS_PER_BATCH = configuration['detectron']['batch_size']
    cfg.SOLVER.BASE_LR = configuration['detectron']['base_learning_rate']
    cfg.SOLVER.STEPS = []

    cfg.TEST.AUG.ENABLED = False
    cfg.TEST.EVAL_PERIOD = configuration['detectron']['eval_period']

    cfg.OUTPUT_DIR = f'./model_runs/{run_name}'

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "validation_evaluation"), exist_ok=True)
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "test_evaluation"), exist_ok=True)

    setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "training_log.txt"))
    
    for tag in condence_tags(cfg):
        mlflow.log_param(tag[0], tag[1])

    trainer = CustomTrainer(cfg)
    trainer.register_hooks(hooks=[MLflowHook(cfg)])
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    with open(os.path.join(cfg.OUTPUT_DIR, "model_config.yaml"), "w") as f:
        f.write(cfg.dump())

    mlflow.log_artifacts(cfg.OUTPUT_DIR)
    
    return cfg


def test_model(cfg, configuration):
    setup_logger(output=os.path.join(cfg.OUTPUT_DIR, "test_evaluation", "evaluation_log.txt"))

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("test", output_dir=os.path.join(cfg.OUTPUT_DIR, "test_evaluation"))
    test_set_loader = build_detection_test_loader(cfg, "test")

    evaluation_results = inference_on_dataset(predictor.model, test_set_loader, evaluator)

    for k, v in evaluation_results["bbox"].items():
        mlflow.log_metric(f"Test {k}", v, step=0)

    mlflow.log_artifacts(os.path.join(cfg.OUTPUT_DIR, "test_evaluation"))
    mlflow.log_text(str(evaluation_results), "test_evaluation/coco_metrics.txt")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    
    config_path = parser.parse_args().config
    
    # load configuration from yaml file
    with open(config_path, 'r') as file:
        configuration = yaml.safe_load(file)

    logging.info(f"Configuration: {configuration}")
    
    mlflow.set_tracking_uri(configuration['mlflow']['tracking_uri'])
    mlflow.set_experiment(configuration['mlflow']['experiment_name'])
    
    register_datasets(configuration)
    
    run_name = time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"Starting Experiment: '{configuration['mlflow']['experiment_name']}' run: '{run_name}'")
    
    mlflow.start_run(run_name=run_name)
    mlflow.set_tag("mlflow.note.content", configuration['mlflow']['run_description'])
    for tag in configuration['mlflow']['tags']:
        mlflow.set_tag(tag['name'], tag['value'])
    
    cfg = train_model(run_name, configuration)
    
    test_model(cfg, configuration)
    
    mlflow.end_run()