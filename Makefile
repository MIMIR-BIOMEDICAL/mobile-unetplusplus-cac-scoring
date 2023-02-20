

download\:dataset:
	python src/data/downloader.py

preprocess\:segment:
	python src/data/preprocess/pipeline/segmentation.py
	
preprocess\:image:
	 if [ -z "$(BATCH)" ]; then \
		python src/data/preprocess/pipeline/image.py;\
	else \
		python src/data/preprocess/pipeline/image.py -b $(BATCH);\
	fi
