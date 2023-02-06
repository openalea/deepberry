## Deepberry : segmentation and time-lapse tracking of grapevine berries

### Installation from source

    # for prediction
    conda create -n grapevine -c conda-forge python=3.7 tensorflow
    conda activate grapevine
    conda install -c conda-forge -c numpy pandas scipy tensorflow matplotlib shapely
    pip install opencv-contrib-python
    pip install pycpd

    # optionnal, for model training (note - for GPU compatibility, see further guides at the top of the training scripts): 
    pip install -U segmentation-models

    # optionnal, for model validation metrics:
    pip install object_detection_metrics

### Maintainers

* Benoit Daviet (benoit.daviet@inrae.fr)
* Fournier Christian (christian.fournier@inrae.fr)




