<p align="center">
    <br>
    <img src="https://ms-vscode-remote.gallerycdn.vsassets.io/extensions/ms-vscode-remote/remote-containers/0.279.0/1675955537911/Microsoft.VisualStudio.Services.Icons.Default" width="100"/>
    <br>
<p>
<p align="center">
    <a href="https://github.com/badges/shields/generate">
        <img src="https://img.shields.io/badge/use%20this-template-blue?logo=github">
    </a>
    <a href="https://github.com/dennisbappert/ml-in-devcontainers/blob/main/LICENSE">
        <img alt="GitHub" src="https://img.shields.io/badge/license-MIT-blue?color=red">
    </a>
</p>

<h3 align="center">
    <p>Immutable development environments for PyTorch powered by Visual Studio Code Dev Containers</p>
</h3>

#

This project provides a development environment for PyTorch using Visual Studio Code and Docker. It is based on the [Visual Studio Code Dev Container](https://github.com/Microsoft/vscode-remote-release) feature.
Being able to have immutable and reproducible development environments is a key part to successfully work with multiple projects.

## Features
- devcontainer.json: Configuration file for the Visual Studio Code Dev Container feature, using a Docker image, based on nvidia/cuda:11.8.0-base-ubuntu22.04
- Optionally: install AWS CLI V2 and reuse your AWS credentials from the host system
- Optionally: install Docker inside the container and reuse your Docker daemon from the host system (if you're training inside a container again)
- Sophisticated set of extensions for Python, Jupyter, etc.
- Sopthisticated set of settings for Python (Linter, Formatter, etc.)
- Sample MNIST training script (PyTorch Example)

## Getting Started
- (Prerequisite) Docker is installed on your system
- (Prerequisite) Visual Studio Code Dev Container extension is installed
- Clone this repository
- Open the folder in Visual Studio Code
- When prompted, click "Reopen in Container"