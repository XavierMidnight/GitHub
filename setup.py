import subprocess

def create_and_activate_environment(env_name):
    """Creates a conda environment and activates it."""
    subprocess.run(["conda", "create", "-n", env_name, "python"], check=True)
    subprocess.run(["conda", "activate", env_name], check=True)

def install_requirements(requirements_file):
    """Installs packages from a requirements file."""
    subprocess.run(["pip", "install", "-r", requirements_file], check=True)

def main():
    env_name = "soundmaker"
    requirements_file = "requirements.txt"  # Replace with your file name

    create_and_activate_environment(env_name)
    install_requirements(requirements_file)
    install_additional_packages = "torch torchaudio einops stable_audio_tools"
    subprocess.run(["pip", "install", install_additional_packages], check=True)

if __name__ == "__main__":
    main()