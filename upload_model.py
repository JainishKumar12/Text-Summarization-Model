from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="saved_summarization_model",
    repo_id="JainishKumar12/text-summarizer-t5",
    repo_type="model"
)