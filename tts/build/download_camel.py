"""
Download required CAMeL Tools models at Docker build time.

This script:
- Ensures catalogue is loaded
- Downloads required packages
- Verifies BERT diacritization model exists
"""

from camel_tools.data.catalogue import Catalogue
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator

print("📦 Loading CAMeL catalogue...")
cat = Catalogue.load_catalogue()

# 🔥 Instead of 'all', download only what you need
# This dramatically reduces image size.
REQUIRED_PACKAGES = [
    "all"
]

for pkg in REQUIRED_PACKAGES:
    print(f"⬇ Downloading package: {pkg}")
    cat.download_package(pkg)

print("🔎 Verifying BERT diacritizer model...")
_ = BERTUnfactoredDisambiguator.pretrained(
    model_name="msa",
    use_gpu=False   # GPU not required during build
)

print("✅ CAMeL models downloaded successfully.")
