## Deepberry : segmentation and time-lapse tracking of grapevine berries

### Installation from source

    # for prediction
    mamba create -n grapevine -c conda-forge python=3.12 (3.9>=version<3.13) 
    mamba activate grapevine

    #(only firts time)
    git lfs install
    git clone
    git lfs pull

    # Install deepberry
    pip install .[test]

    # (optional) for model training (note: for GPU compatibility, see further guidelines at the top of the training scripts): 
    pip install .[training]

    # (optional) for model validation metrics:
    pip install .[validation]

### Maintainers

* Benoit Daviet (b.daviet@hotmail.com)
* Fournier Christian (christian.fournier@inrae.fr)




