# BitsAndBytes Docker (with AMD GPU support)

A Docker environment for running BitsAndBytes with AMD GPU support, specifically configured for ROCm 6.1.2 and PyTorch. This setup is optimized for AMD GPUs with gfx1030 architecture.

## Versions

**Current version**: 1.0.0 - Initial release with ROCm 6.1.2 support and PyTorch integration

## Table of Contents

- [Versions](#versions)
- [Badges](#badges)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contributing](#contributing)

## Badges

![YAML](https://img.shields.io/badge/YAML-Used-blue)
![Docker](https://img.shields.io/badge/Docker-Required-blue)
![Version](https://img.shields.io/badge/version-1.0.0-green)
![ROCm](https://img.shields.io/badge/ROCm-6.1.2-red)
![License](https://img.shields.io/badge/license-MIT-green)

## Installation

1. Ensure you have Docker and Docker Compose installed on your system
2. Clone this repository:
   ```bash
   git clone https://github.com/beecave-homelab/bitsandbytes-rocm-docker.git
   cd bitsandbytes-rocm-docker
   ```
3. Make sure you have an AMD GPU that supports ROCm (specifically gfx1030 architecture)
4. Verify that ROCm is properly installed on your host system

## Usage

1. Build and start the container:
   ```bash
   docker-compose up -d
   ```

2. Access the container shell:
   ```bash
   docker-compose exec rocm-app bash
   ```

The environment includes:
- ROCm 6.1.2 development tools
- PyTorch with ROCm 6.1 support
- BitsAndBytes library compiled with HIP backend
- Python 3 development environment

### Environment Variables

The following environment variables are pre-configured:
- `HSA_OVERRIDE_GFX_VERSION=10.3.0`
- `HCC_AMDGPU_TARGET=gfx1030`
- `HIP_VISIBLE_DEVICES=0`

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE) for more information.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

