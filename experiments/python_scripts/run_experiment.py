# SPDX-FileCopyrightText: 2023 Adam Wojdyła <adam.wojdyla@outlook.com>
#
# SPDX-License-Identifier: MIT

# pylint: disable=duplicate-code, broad-except
import os
import sys

import hydra
import mlflow
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

ROOT_FOLDER_PATH = os.path.abspath(
    os.path.join(os.path.realpath(__file__), "../../../")
)
if ROOT_FOLDER_PATH not in sys.path:
    sys.path.append(ROOT_FOLDER_PATH)

from data import loaders
from internal_modules import azure_workspace, plots
from internal_modules.training import conduct_experiment
from internal_modules.utils import data_utils, mlflow_utils, model_utlis, torch_utils


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch_utils.select_device()

    # Run augmentation
    if cfg.get("augmentation", None) is not None:
        data_utils.run_augmentation(
            subset_size=cfg.dataset.subset_size,
            source_dir=os.path.join(ROOT_FOLDER_PATH, cfg.paths.dataset),
            dest_dir=os.path.join(ROOT_FOLDER_PATH, cfg.paths.dataset_augmented),
            augmentation=cfg.augmentation,
        )
        cfg.paths.dataset = cfg.paths.dataset_augmented
        if cfg.dataset.subset_size is not None:
            cfg.dataset.subset_size *= 2

    # Load data
    data_loader = loaders.FlatsDatasetLoader(
        images_dir=os.path.join(ROOT_FOLDER_PATH, cfg.paths.dataset),
        resize_to=cfg.dataset.image_size,
        device=device,
        batch_size=cfg.params.batch_size,
    )
    full_dataset = data_loader.load(
        verbose=True,
        subset_size=cfg.dataset.subset_size,
        load_to_ram=cfg.dataset.load_to_ram,
    )
    classes = list(data_loader.get_label_names().values())
    print("Dataset description: ", data_loader)

    # Prepare model
    model = instantiate(cfg.model.target)
    model = model_utlis.freeze_model(
        model, cfg.model.name, data_loader.get_classes_count(), device
    )
    optimizer = model_utlis.get_optimizer(model, cfg)

    # Run model training
    print(f"Training: {cfg.model.description} started on device: {device}")

    if not cfg.mlflow.mlflow_enabled:
        conduct_experiment(
            title=cfg.model.description,
            model=model,
            n_epochs=cfg.params.epoch_count,
            optimizer=optimizer,
            flats_dataset_loader=data_loader,
            device=device,
            register_model=cfg.mlflow.register_model,
        )
    else:
        try:
            if cfg.mlflow.mlflow_azure_upstream:
                print("Setting MLflow experiment to Azure workspace")
                azure_workspace.set_mlflow_experiment(cfg)
            else:
                print("Setting MLflow experiment to local")
            mlflow.set_experiment(cfg.model.description)
            mlflow.start_run()
            print(
                "MLflow run started. "
                "Evaluation results will be logged iteratively after each epoch. "
                "To view metrics, run 'mlflow ui' in the terminal."
            )
            mlflow_utils.log_run_mlflow(
                full_dataset=full_dataset,
                dataset_info=data_loader.as_dict(),
                device=device,
                augmentations=None,
                config=cfg,
            )
            conduct_experiment(
                title=cfg.model.description,
                model=model,
                n_epochs=cfg.params.epoch_count,
                optimizer=optimizer,
                flats_dataset_loader=data_loader,
                device=device,
                use_mlflow=cfg.mlflow.mlflow_enabled,
                register_model=cfg.mlflow.register_model,
            )
            plots_dict = plots.create_plots_from_experiment(
                test_dataset=data_loader.get_test_loader(),
                full_dataset=full_dataset,
                dataset_subset=cfg.dataset.subset_size,
                classes=classes,
                model=model,
                device=device,
            )
            mlflow_utils.log_plots_mlflow(plots_dict)
            print("Experiment finished successfully.")
        except Exception as fatal_error:
            print(f"Unexpected error while running training: {fatal_error}")
            print(f"Ending run. Run info: {mlflow.active_run().info}")
        finally:
            mlflow.end_run()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
    # pylint: enable=no-value-for-parameter
