import os

import requests


def get_latest_version(package_name):
    response = requests.get(f"https://pypi.org/pypi/{package_name}/json")
    data = response.json()
    return data["info"]["version"]


def read_requirements(file_path):
    with open(file_path) as file:
        lines = file.readlines()

    requirements = {}
    for line in lines:
        if "==" in line:
            package, version = line.strip().split("==")
            requirements[package.split("[")[0]] = version.split("#")[0].strip()

    return requirements


def main():
    path = os.path.abspath("requirements")
    dir_list = os.listdir(path)
    for file in dir_list:
        requirements = read_requirements(f"{path}/{file}")
        for package, version in requirements.items():
            latest_version = get_latest_version(package)
            if version != latest_version:
                print(f"{package} is not up-to-date. Current version: {version}, latest version: {latest_version}")


if __name__ == "__main__":
    main()
