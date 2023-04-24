import evaluate
from konlpy.tag import Komoran

komoran = Komoran()
rouge = evaluate.load("rouge")


def rouge_for_sampale(label, prediction):
    return rouge.compute(
        references=[label], predictions=[prediction], tokenizer=komoran.morphs
    )


def rouge_for_batch(labels, predictions):
    rouge_scores = None

    for label, prediction in zip(labels, predictions):
        if rouge_scores == None:
            rouge_scores = rouge_for_sampale(label, prediction)
        else:
            rouge_score = rouge_for_sampale(label, prediction)
            for key in rouge_scores.keys():
                rouge_scores[key] = rouge_scores[key] + rouge_score[key]

    for key in rouge_scores.keys():
        rouge_scores[key] = rouge_scores[key] / len(labels)

    return rouge_scores


def f1_score_at_k_for_sample(label_str, prediction_str, k):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    label_lst = [
        key_phrase.strip() for key_phrase in label_str.split(";") if key_phrase != ""
    ]
    label_lst = [key_phrase for key_phrase in label_lst if key_phrase != ""]

    prediction_lst = [
        key_phrase.strip()
        for key_phrase in prediction_str.split(";")
        if key_phrase != ""
    ]
    prediction_lst = [key_phrase for key_phrase in prediction_lst if key_phrase != ""][
        :k
    ]

    for keyphrase in prediction_lst:
        similarity = False
        for label in label_lst:
            if keyphrase in label or label in keyphrase:
                similarity = True
                break
        if similarity == True:
            true_positives += 1
        else:
            false_positives += 1

    for label in label_lst:
        similarity = False
        for keyphrase in prediction_lst:
            if label in keyphrase or keyphrase in label:
                similarity = True
                break
        if similarity == False:
            false_negatives += 1

    # calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision == 0 or recall == 0:
        return 0

    f1_score = 2 * (precision * recall) / (precision + recall)

    return f1_score


def f1_score_at_k_for_batch(labels, predictions, k):
    f1_scores = []

    for label, prediction in zip(labels, predictions):
        f1_scores.append(f1_score_at_k_for_sample(label, prediction, k))

    return sum(f1_scores) / len(f1_scores)


def jaccard_similarity_for_sample(label, prediction, k):
    label_lst = [
        key_phrase.strip() for key_phrase in label.split(";") if key_phrase != ""
    ]
    label_lst = [key_phrase for key_phrase in label_lst if key_phrase != ""]

    prediction_lst = [
        key_phrase.strip() for key_phrase in prediction.split(";") if key_phrase != ""
    ]
    prediction_lst = [key_phrase for key_phrase in prediction_lst if key_phrase != ""][
        :k
    ]

    """Define Jaccard Similarity function for two sets"""
    intersection = len(list(set(label_lst).intersection(prediction_lst)))
    union = (len(label_lst) + len(prediction_lst)) - intersection

    return float(intersection) / union


def jaccard_similarity_for_batch(labels, predictions, k):
    jaccard_similarities = []

    for label, prediction in zip(labels, predictions):
        jaccard_similarities.append(jaccard_similarity_for_sample(label, prediction, k))

    return sum(jaccard_similarities) / len(jaccard_similarities)
