from huggingface_hub import snapshot_download
import os

# Download only the lesion_segmentation folder to data/ddr_seg/
snapshot_download(
    repo_id="ctmedtech/DDR-dataset",
    repo_type="dataset",
    allow_patterns="lesion_segmentation/*",
    local_dir=os.path.join("data", "ddr_seg"),
    local_dir_use_symlinks=False,
)

print("Download complete! Check data/ddr_seg/lesion_segmentation/")
