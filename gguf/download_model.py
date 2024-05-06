from huggingface_hub import snapshot_download
import argparse

def main():
    parser = argparse.ArgumentParser(description="Downloads a model from HF hub to disk.")
    parser.add_argument('model_name', type=str, 
                        help='The name of the model to download.')
    parser.add_argument('folder', type=str, 
                        help='The output folder to store the model.')
    

    args = parser.parse_args()
    model_name = args.model_name
    folder = args.folder

    snapshot_download(repo_id=model_name, local_dir=folder, local_dir_use_symlinks=False, revision="main")

if __name__ == "__main__":
    main()
