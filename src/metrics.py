def f1_score_at_k_for_sample(label, prediction, k):
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # convert label and prediction strings to sets of key-phrases
    label_lst = [key_phrase.strip() for key_phrase in label.split(';') if key_phrase != '']
    label_lst = [key_phrase for key_phrase in label_lst if key_phrase != '']
    label_set = set(label_lst)
    
    # split the predicted key-phrases and their scores
    prediction_lst = [key_phrase.strip() for key_phrase in prediction.split(';') if key_phrase != '']
    prediction_lst = [key_phrase for key_phrase in prediction_lst if key_phrase != ''][:k]
    prediction_set = set(prediction_lst)
    
    # calculate true positives, false positives, and false negatives
    for keyphrase in prediction_set:
        if keyphrase in label_set:
            true_positives += 1
        else:
            false_positives += 1
    
    for keyphrase in label_set:
        if keyphrase not in prediction_set:
            false_negatives += 1

    # calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    
    if precision == 0 or recall == 0:
        return 0
    
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score

def f1_score_at_k_for_batch(labels, predictions, k):
    f1_scores =[]

    for label, prediction in zip(labels, predictions):
        f1_scores.append(f1_score_at_k_for_sample(label, prediction, k))

    print(f1_scores)
    return sum(f1_scores) / len(f1_scores)

def jaccard_similarity_for_sample(label, prediction, k):

    # convert label and prediction strings to sets of key-phrases
    label_lst = [key_phrase.strip() for key_phrase in label.split(';') if key_phrase != '']
    label_lst = [key_phrase for key_phrase in label_lst if key_phrase != '']
    
    # split the predicted key-phrases and their scores
    prediction_lst = [key_phrase.strip() for key_phrase in prediction.split(';') if key_phrase != '']
    prediction_lst = [key_phrase for key_phrase in prediction_lst if key_phrase != ''][:k]

    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(label_lst).intersection(prediction_lst)))
    union = (len(label_lst) + len(prediction_lst)) - intersection

    return float(intersection) / union

def jaccard_similarity_for_batch(labels, predictions, k):
    jaccard_similarities =[]

    for label, prediction in zip(labels, predictions):
        jaccard_similarities.append(jaccard_similarity_for_sample(label, prediction, k))

    print(jaccard_similarities)
    return sum(jaccard_similarities) / len(jaccard_similarities)