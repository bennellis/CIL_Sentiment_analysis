import subprocess
import inspect
import os
import functools
import mlflow


def get_git_output(command_list):
    try:
        return subprocess.check_output(command_list).decode().strip()
    except subprocess.CalledProcessError:
        return None


def get_class_file_path(model_class):
    return os.path.abspath(inspect.getfile(model_class))


def is_file_dirty(file_path):
    output = get_git_output(["git", "status", "--porcelain", file_path])
    return bool(output)


def get_git_info():
    return {
        "user_name": get_git_output(["git", "config", "user.name"]),
        "user_email": get_git_output(["git", "config", "user.email"]),
        "commit_sha": get_git_output(["git", "rev-parse", "HEAD"]),
        "branch": get_git_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "remote_url": get_git_output(["git", "config", "--get", "remote.origin.url"]),
    }


def get_github_link(remote_url, file_path, commit_sha):
    if not remote_url or not commit_sha:
        return None

    if remote_url.endswith(".git"):
        remote_url = remote_url[:-4]

    # Convert SSH URL to HTTPS
    if remote_url.startswith("git@github.com:"):
        remote_url = remote_url.replace("git@github.com:", "https://github.com/")

    # Get relative file path to repo root
    repo_root = get_git_output(["git", "rev-parse", "--show-toplevel"])
    relative_path = os.path.relpath(file_path, repo_root)

    return f"{remote_url}/blob/{commit_sha}/{relative_path}"
def log_model_git_info(name, registry_info):
    git_info = registry_info.get("git_info")
    file_path = registry_info.get("file_path")
    was_dirty = registry_info.get("dirty")

    file_link = get_github_link(git_info["remote_url"], file_path, git_info["commit_sha"])
    mlflow.set_tag("model_name", name)
    mlflow.set_tag("model_file", file_path)
    mlflow.set_tag("git_commit", git_info["commit_sha"])
    mlflow.set_tag("git_user", f"{git_info['user_name']} <{git_info['user_email']}>")
    mlflow.set_tag("git_branch", git_info["branch"])
    mlflow.set_tag("model_file_link", file_link)
    mlflow.set_tag("dirty", was_dirty)


def log_embedding_git_info(name, registry_info):
    git_info = registry_info.get("git_info")
    file_path = registry_info.get("file_path")
    was_dirty = registry_info.get("dirty")

    file_link = get_github_link(git_info["remote_url"], file_path, git_info["commit_sha"])
    mlflow.set_tag("embedding_name", name)
    mlflow.set_tag("embedding_file", file_path)
    mlflow.set_tag("git_commit", git_info["commit_sha"])
    mlflow.set_tag("git_user", f"{git_info['user_name']} <{git_info['user_email']}>")
    mlflow.set_tag("git_branch", git_info["branch"])
    mlflow.set_tag("embedding_file_link", file_link)
    mlflow.set_tag("dirty", was_dirty)