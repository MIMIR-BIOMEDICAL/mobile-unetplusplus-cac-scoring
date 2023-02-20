BATCH ?=
SAMPLE ?=

download\:dataset:
	python src/data/downloader.py

preprocess\:segment:
	python src/data/preprocess/pipeline/segmentation.py
	
preprocess\:image:
	python src/data/preprocess/pipeline/image.py \
        $(if $(BATCH),-b $(BATCH)) \
        $(if $(SAMPLE),-s $(SAMPLE))

