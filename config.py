class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    num_final_features = 12 # 1 + 11 (one hot representation)
    # for testing (one of the categories wasn't seen)
    # num_final_features = 11

    batch_size = 64
    num_classes = 20
    # for testing (one of the categories wasn't seen)
    # num_classes = 16
    num_hidden = 40

    num_layers = 3

    num_epochs = 50
    l2_lambda = 0.0000001
    lr = 1e-3

    classifier_lr = 0.1
    classifier_l1_lambda = 1.0
    classifier_l2_lambda = 1.0