BATCH ?=
SAMPLE ?=
SPLIT_TYPE ?=
SPLIT_DISTRIBUTION ?=

download\:dataset:
	python src/data/downloader.py

preprocess\:segment:
	python src/data/preprocess/pipeline/segmentation.py
	
preprocess\:image:
	python src/data/preprocess/pipeline/image.py \
        $(if $(BATCH),-b $(BATCH)) \
        $(if $(SAMPLE),-s $(SAMPLE))

preprocess\:tfrecord:
	python src/data/preprocess/pipeline/tfrecord.py \
        $(if $(SAMPLE),-s $(SAMPLE)) \
        $(if $(SPLIT_TYPE),-t $(SPLIT_TYPE)) \
        $(if $(SPLIT_DISTRIBUTION),-d $(SPLIT_DISTRIBUTION)) 
		

