# Docker for SimpleTuner

This Docker configuration provides a comprehensive environment for running the SimpleTuner application on various platforms including Runpod, Vast.ai, and other Docker-compatible hosts. It is optimized for ease of use and robustness, integrating tools and libraries essential for machine learning projects.

## Container Features

- **CUDA-enabled Base Image**: Built from `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` to support GPU-accelerated applications.
- **Development Tools**: Includes Git, SSH, and various utilities like `tmux`, `vim`, `htop`.
- **Python and Libraries**: Comes with Python 3.10 and SimpleTuner pre-installed via pip.
- **Huggingface and WandB Integration**: Pre-configured for seamless integration with Huggingface Hub and WandB, facilitating model sharing and experiment tracking.

## Getting Started

### Windows OS support via WSL (Experimental)

The following guide was tested in a WSL2 Distro that has Dockerengine installed.


### 1. Building the Container

Clone the repository and navigate to the directory containing the Dockerfile. Build the Docker image using:

```bash
docker build -t simpletuner .
```

### 2. Running the Container

To run the container with GPU support, execute:

```bash
docker run --gpus all -it -p 22:22 simpletuner
```

This command sets up the container with GPU access and maps the SSH port for external connectivity.

### 3. Environment Variables

To facilitate integration with external tools, the container supports environment variables for Huggingface and WandB tokens. Pass these at runtime as follows:

```bash
docker run --gpus all -e HF_TOKEN='your_token' -e WANDB_API_KEY='your_token' -it -p 22:22 simpletuner
```

### 4. Data Volumes

For persistent storage and data sharing between the host and the container, mount a data volume:

```bash
docker run --gpus all -v /path/on/host:/workspace -it -p 22:22 simpletuner
```

### 5. SSH Access

SSH into the container is configured by default. Ensure you provide your SSH public key through the appropriate environment variable (`SSH_PUBLIC_KEY` for Vast.ai or `PUBLIC_KEY` for Runpod).

### 6. Using SimpleTuner

SimpleTuner is pre-installed and ready to use. You can run training commands directly:

```bash
simpletuner configure
simpletuner train
```

For configuration and setup, refer to the [installation documentation](INSTALL.md) and [quickstart guides](QUICKSTART.md).

## Additional Configuration

### Custom Scripts and Configurations

If you want to add custom startup scripts or modify configurations, extend the entry script (`docker-start.sh`) to fit your specific needs.

If any capabilities cannot be achieved through this setup, please open a new issue.

### Docker Compose

For users who prefer `docker-compose.yaml`, this template is provided for you to extend and customise for your needs.

Once the stack is deployed you can connect to the container and start operating in it as mentioned in the steps above.

```bash
docker compose up -d

docker exec -it simpletuner /bin/bash
```

```docker-compose.yaml
services:
  simpletuner:
    container_name: simpletuner
    build:
      context: [Path to the repository]/SimpleTuner
      dockerfile: Dockerfile
    ports:
      - "[port to connect to the container]:22"
    volumes:
      - "[path to your datasets]:/datasets"
      - "[path to your configs]:/workspace/config"
    environment:
      HF_TOKEN: [your hugging face token]
      WANDB_API_KEY: [your wanddb token]
    command: ["tail", "-f", "/dev/null"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

> ⚠️ Please be cautious of handling your WandB and Hugging Face tokens! It's advised not to commit them even to a private version-control repository to ensure they are not leaked. For production use-cases, key management storage is recommended, but out of scope for this guide.
---

## Troubleshooting

### CUDA Version Mismatch

**Symptom**: The application fails to utilize the GPU, or errors related to CUDA libraries appear when attempting to run GPU-accelerated tasks.

**Cause**: This issue may occur if the CUDA version installed within the Docker container does not match the CUDA driver version available on the host machine.

**Solution**:
1. **Check CUDA Driver Version on Host**: Determine the version of the CUDA driver installed on the host machine by running:
   ```bash
   nvidia-smi
   ```
   This command will display the CUDA version at the top right of the output.

2. **Match Container CUDA Version**: Ensure that the version of the CUDA toolkit in your Docker image is compatible with the host's CUDA driver. NVIDIA generally allows forward compatibility but check the specific compatibility matrix on the NVIDIA website.

3. **Rebuild the Image**: If necessary, modify the base image in the Dockerfile to match the host’s CUDA driver. For example, if your host runs CUDA 11.2 and the container is set up for CUDA 11.8, you might need to switch to an appropriate base image:
   ```Dockerfile
   FROM nvidia/cuda:11.2.0-runtime-ubuntu22.04
   ```
   After modifying the Dockerfile, rebuild the Docker image.

### SSH Connection Issues

**Symptom**: Unable to connect to the container via SSH.

**Cause**: Misconfiguration of SSH keys or the SSH service not starting correctly.

**Solution**:
1. **Check SSH Configuration**: Ensure that the public SSH key is correctly added to `~/.ssh/authorized_keys` in the container. Also, verify that the SSH service is up and running by entering the container and executing:
   ```bash
   service ssh status
   ```
2. **Exposed Ports**: Confirm that the SSH port (22) is properly exposed and mapped when starting the container, as shown in the running instructions:
   ```bash
   docker run --gpus all -it -p 22:22 simpletuner
   ```

### General Advice

- **Logs and Output**: Review the container logs and output for any error messages or warnings that can provide more context on the issue.
- **Documentation and Forums**: Consult the Docker and NVIDIA CUDA documentation for more detailed troubleshooting advice. Community forums and issue trackers related to the specific software or dependencies you are using can also be valuable resources.
