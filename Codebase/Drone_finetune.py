#pi0_drone_finetune.py
#This script is used to finetune the pi0_drone model on the Cognitive Drone dataset
# Clone the CognitiveDrone dataset from Hugging Face with "git clone https://huggingface.co/datasets/ArtemLykov/CognitiveDrone_dataset
# Save the dataset to the /mnt/storage/drone_dataset directory within the storage volume: drone-training-data
# Clone the OpenPI repository with "git clone https://github.com/Physical-Intelligence/openpi.git"
# Install the OpenPI package with "cd openpi && pip install -e ."

import os
import sys
import time
import logging
import shutil
import json
import glob
from pathlib import Path
from datetime import datetime

# Modal imports
import modal

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the Modal image with necessary dependencies
image = modal.Image.debian_slim().pip_install(
    "torch==2.1.0",
    "accelerate",
    "transformers",
    "datasets",
    "numpy",
    "pillow",
    "tqdm",
    "scipy",
    "tensorboard",
    "matplotlib",
    "opencv-python-headless",
    "safetensors",
    "wandb",
    "diffusers>=0.21.0",
    "xformers",
    "bitsandbytes",
    "pandas",
    "einops"
)
# Build the container image with dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libosmesa6-dev", "libgl1-mesa-glx", "libglew-dev", "libglfw3-dev", "libgles2-mesa-dev", "git"])
    .pip_install([
        # Core ML/robotics libraries
        "numpy", "flax", "dm-haiku", "gin-config", "websockets", 
        # JAX with CUDA support
        "jax==0.5.1",
        "jaxlib==0.5.1",
    ])
    .run_commands([
        # Upgrade pip
        "python -m pip install --upgrade pip",
        # Install other dependencies
        "pip install --upgrade jax[cuda12_pip] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html",
        "git clone https://github.com/Physical-Intelligence/openpi.git /openpi",
        # Install openpi-client package (provides client utilities, and may include server dependencies)
        "pip install -e /openpi/packages/openpi-client",
    ])
)

# Create a storage volume for the dataset and model weights
storage_volume = modal.Volume.from_name("drone-training-data", create_if_missing=True)
model_volume = modal.Volume.from_name("pi0-model-weights", create_if_missing=True)

# Create the Modal app
app = modal.App("pi0-drone-finetune", image=image)

def download_dataset():
    """
    Download the CognitiveDrone dataset if not already present
    """
    dataset_path = "/mnt/storage/drone_dataset"
    if not os.path.exists(dataset_path):
        logger.info("Downloading CognitiveDrone dataset...")
        os.makedirs(dataset_path, exist_ok=True)
        os.system(f"git clone https://huggingface.co/datasets/ArtemLykov/CognitiveDrone_dataset {dataset_path}")
        logger.info("Dataset downloaded successfully")
    else:
        logger.info("Dataset already exists, skipping download")
    
    # List the dataset structure
    logger.info("Dataset structure:")
    os.system(f"find {dataset_path} -type f | sort | head -20")

def preprocess_drone_dataset(dataset_path, output_path):
    """
    Preprocess the drone dataset into the format expected by the π₀ model
    """
    logger.info(f"Preprocessing dataset from {dataset_path} to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # For demonstration, we'll assume the dataset contains:
    # - Images in a folder structure
    # - JSON annotations with action data
    
    # Find all JSON annotation files
    annotation_files = glob.glob(f"{dataset_path}/**/*.json", recursive=True)
    logger.info(f"Found {len(annotation_files)} annotation files")
    
    # Process each annotation file
    for i, annotation_file in enumerate(annotation_files):
        logger.info(f"Processing annotation {i+1}/{len(annotation_files)}: {annotation_file}")
        
        with open(annotation_file, 'r') as f:
            data = json.load(f)
        
        # Convert to format expected by π₀ model - this will need to be adapted
        # based on the actual structure of the CognitiveDrone dataset
        sample = {
            "id": f"drone_{i}",
            "observation": {
                "exterior_image_1_left": data.get("drone_image", ""),
                # Add other observation keys as needed
            },
            "prompt": data.get("instruction", "Navigate the drone to reach the target"),
            "actions": data.get("actions", []),
            # Add other fields required by π₀
        }
        
        # Save processed sample
        output_file = os.path.join(output_path, f"sample_{i}.json")
        with open(output_file, 'w') as f:
            json.dump(sample, f, indent=2)
    
    logger.info(f"Preprocessing complete. Created {len(annotation_files)} samples.")

@app.function(
    gpu="A100",
    timeout=86400,  # 24 hours
    volumes={
        "/mnt/storage": storage_volume,
        "/mnt/models": model_volume,
    }
)
def finetune_pi0_model():
    """
    Finetune the π₀ model on the drone dataset
    """
    logger.info("Starting finetuning process")
    
    # Set working directory to OpenPI
    os.chdir("/openpi")
    
    # Download the dataset
    download_dataset()
    
    # Set paths
    dataset_path = "/mnt/storage/drone_dataset"
    processed_data_path = "/mnt/storage/processed_drone_data"
    output_dir = "/mnt/models/pi0_drone_finetuned"
    
    # Preprocess the dataset
    preprocess_drone_dataset(dataset_path, processed_data_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure training parameters
    base_model = "s3://openpi-assets/checkpoints/pi0_base"
    
    # Create training configuration
    config_file = "/tmp/pi0_drone_config.py"
    with open(config_file, "w") as f:
        f.write("""
import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Model configuration
    config.model = ml_collections.ConfigDict()
    config.model.name = "pi0"
    config.model.base_checkpoint = "s3://openpi-assets/checkpoints/pi0_base"
    
    # Dataset configuration
    config.dataset = ml_collections.ConfigDict()
    config.dataset.name = "drone"
    config.dataset.path = "/mnt/storage/processed_drone_data"
    config.dataset.train_split = "train"
    config.dataset.val_split = "validation"
    
    # Training configuration
    config.training = ml_collections.ConfigDict()
    config.training.batch_size = 2
    config.training.gradient_accumulation_steps = 4
    config.training.learning_rate = 1e-5
    config.training.weight_decay = 0.01
    config.training.num_train_steps = 20000
    config.training.checkpoint_every = 1000
    config.training.eval_every = 500
    config.training.seed = 42
    
    # LoRA configuration for efficient fine-tuning
    config.lora = ml_collections.ConfigDict()
    config.lora.enabled = True
    config.lora.r = 16
    config.lora.alpha = 32
    config.lora.dropout = 0.1
    config.lora.target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
    
    # Optimization
    config.optimizer = ml_collections.ConfigDict()
    config.optimizer.name = "adamw"
    config.optimizer.beta1 = 0.9
    config.optimizer.beta2 = 0.999
    config.optimizer.eps = 1e-8
    
    # Scheduler
    config.scheduler = ml_collections.ConfigDict()
    config.scheduler.name = "cosine"
    config.scheduler.warmup_steps = 1000
    
    # Hardware
    config.hardware = ml_collections.ConfigDict()
    config.hardware.mixed_precision = True
    config.hardware.fsdp_devices = 1  # Number of GPUs to use
    
    # Logging
    config.logging = ml_collections.ConfigDict()
    config.logging.log_every = 50
    config.logging.use_wandb = False
    
    # Output
    config.output_dir = "/mnt/models/pi0_drone_finetuned"
    
    return config
""")
    
    # Start training
    logger.info("Starting training with the following command:")
    train_cmd = f"python -m openpi.training.train --config={config_file} --workdir={output_dir}"
    logger.info(train_cmd)
    
    # Execute training
    os.system(train_cmd)
    
    # Verify training results
    if os.path.exists(os.path.join(output_dir, "final_checkpoint")):
        logger.info("Training completed successfully!")
        # List the model files
        logger.info("Generated model files:")
        os.system(f"find {output_dir} -type f | sort")
    else:
        logger.error("Training failed or did not complete")
    
    # Create a README file with training information
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(f"""# Finetuned π₀ Model for Drone Control

This model was finetuned on the CognitiveDrone dataset.

- Base model: Physical Intelligence π₀ diffusion model
- Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Dataset: CognitiveDrone

## Usage

```python
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_drone")
checkpoint_dir = "/path/to/this/directory"

# Create a trained policy
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a drone observation
example = {{
    "observation/exterior_image_1_left": drone_image,
    "prompt": "Navigate to the red marker"
}}
action_chunk = policy.infer(example)["actions"]
```
""")
    
    # Commit changes to the volume to ensure they are saved
    storage_volume.commit()
    model_volume.commit()
    
    return os.path.join(output_dir, "final_checkpoint")

@app.function(
    gpu="A100", 
    volumes={
        "/mnt/models": model_volume,
    }
)
def test_finetuned_model(checkpoint_path):
    """
    Test the finetuned model with a sample input
    """
    logger.info(f"Testing model from checkpoint: {checkpoint_path}")
    
    # Set working directory to OpenPI
    os.chdir("/openpi")
    
    # Create a sample test script
    test_script = """
from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Create a dummy test image (in practice, use a real drone image)
def create_test_image(width=224, height=224):
    # Create a random image with a simple pattern
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Add a diagonal line
    for i in range(min(width, height)):
        img[i, i] = [255, 0, 0]  # Red diagonal
    # Add a target point
    center_x, center_y = width // 2, height // 2
    for x in range(center_x-5, center_x+5):
        for y in range(center_y-5, center_y+5):
            if 0 <= x < width and 0 <= y < height:
                img[y, x] = [0, 255, 0]  # Green target
    return Image.fromarray(img)

# Load the finetuned model
config_path = "/tmp/pi0_drone_config.py"
checkpoint_dir = "{checkpoint_path}"

# Get the config
import importlib.util
import sys
spec = importlib.util.spec_from_file_location("config_module", config_path)
config_module = importlib.util.module_from_spec(spec)
sys.modules["config_module"] = config_module
spec.loader.exec_module(config_module)
config = config_module.get_config()

# Create a trained policy
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Create a test image
test_image = create_test_image()
test_image.save("/tmp/test_drone_image.png")
print("Created test image at /tmp/test_drone_image.png")

# Convert to the format expected by the model
test_image_array = np.array(test_image)
print(f"Test image shape: {test_image_array.shape}")

# Run inference
example = {{
    "observation/exterior_image_1_left": test_image_array,
    "prompt": "Navigate the drone to reach the green target"
}}

print("Running inference...")
result = policy.infer(example)
print("Inference completed")
print(f"Actions shape: {result['actions'].shape}")
print(f"Action sample: {result['actions'][0]}")
print("Test completed successfully!")
"""
    
    # Save the test script
    with open("/tmp/test_pi0_drone.py", "w") as f:
        f.write(test_script.format(checkpoint_path=checkpoint_path))
    
    # Run the test script
    logger.info("Running test script...")
    os.system("python /tmp/test_pi0_drone.py")
    
    return "Model testing completed"

@app.function()
def main():
    """
    Main function to orchestrate the finetuning process
    """
    logger.info("Starting π₀ model finetuning for drone data")
    
    # Start the finetuning process
    checkpoint_path = finetune_pi0_model.remote()
    
    # Test the finetuned model
    test_result = test_finetuned_model.remote(checkpoint_path)
    
    logger.info(f"Finetuning completed. Model saved to: {checkpoint_path}")
    logger.info(f"Test result: {test_result}")
    
    return {
        "status": "success",
        "checkpoint_path": checkpoint_path,
        "test_result": test_result
    }

if __name__ == "__main__":
    # When running locally as a script, call modal.run directly
    modal.run(main)