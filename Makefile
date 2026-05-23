.PHONY: download_artifacts

ZENODO_RECORD_ID = XXXXXXX
DATA_ZENODO_URL = "https://zenodo.org/record/$(ZENODO_RECORD_ID)/files/data.zip"
RESULTS_ZENODO_URL = "https://zenodo.org/record/$(ZENODO_RECORD_ID)/files/results.zip"

all: download_artifacts

download_artifacts:
	@echo "=== Downloading data/ and results/ folders from Zenodo ==="
	@mkdir -p data results
	curl -L -o data.zip $(DATA_ZENODO_URL)
	curl -L -o results.zip $(RESULTS_ZENODO_URL)
	unzip -o data.zip -d .
	unzip -o results.zip -d .
	rm data.zip results.zip
	@echo "=== Artifacts successfully downloaded and extracted ==="