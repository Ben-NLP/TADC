class args():
    hidden_size = 300
    topic_num = 24  # 22 for laptop, 24 for restaurant
    shortcut = True
    top_k = 10

    dropout = 0.55

    lr = 0.0001
    epochs = 300  # 400 for laptop, 300 for restaurant
    num_valid = 150
    batch_size = 128
    
    data_dir = 'data/prep_data/'
    model_dir = 'model/'