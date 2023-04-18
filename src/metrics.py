def f1_score_at_k(label, prediction, k):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # convert label and prediction strings to sets of key-phrases
    label_lst = [key_phrase.strip() for key_phrase in label.split(';') if key_phrase != '']
    label_lst = [key_phrase for key_phrase in label_lst if key_phrase != '']
    label_set = set(label_lst)
    print(f'label_set: {label_set}')
    
    # split the predicted key-phrases and their scores
    prediction_lst = [key_phrase.strip() for key_phrase in prediction.split(';') if key_phrase != '']
    prediction_lst = [key_phrase for key_phrase in prediction_lst if key_phrase != ''][:k]
    prediction_set = set(prediction_lst)
    # prediction_set = set(p[0] for p in predictions[:k])
    print(f'prediction_set: {prediction_set}')
    
    # calculate true positives, false positives, and false negatives
    for keyphrase in prediction_set:
        if keyphrase in label_set:
            true_positives += 1
        else:
            false_positives += 1
    
    for keyphrase in label_set:
        if keyphrase not in prediction_set:
            false_negatives += 1
    
    print(f'true_positives: {true_positives}')    
    print(f'false_positives: {false_positives}')
    print(f'false_negatives: {false_negatives}')

    # calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision == 0 or recall == 0:
        return 0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score