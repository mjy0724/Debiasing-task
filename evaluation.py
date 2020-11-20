import numpy as np


def evaluate_each_file(predictions, answers):
    list_item_degress = []
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        list_item_degress.append(item_degree)
    list_item_degress.sort()
    median_item_degree = list_item_degress[len(list_item_degress) // 2]

    num_cases_full = 0.0
    ndcg_50_full = 0.0
    ndcg_50_half = 0.0
    num_cases_half = 0.0
    hitrate_50_full = 0.0
    hitrate_50_half = 0.0
    for user_id in answers:
        item_id, item_degree = answers[user_id]
        rank = 0
        while rank < 50 and predictions[user_id][rank] != item_id:
            rank += 1
        num_cases_full += 1.0
        if rank < 50:
            ndcg_50_full += 1.0 / np.log2(rank + 2.0)
            hitrate_50_full += 1.0
        if item_degree <= median_item_degree:
            num_cases_half += 1.0
            if rank < 50:
                ndcg_50_half += 1.0 / np.log2(rank + 2.0)
                hitrate_50_half += 1.0
    ndcg_50_full /= num_cases_full
    hitrate_50_full /= num_cases_full
    ndcg_50_half /= num_cases_half
    hitrate_50_half /= num_cases_half
    return np.array([ndcg_50_full, ndcg_50_half,
                     hitrate_50_full, hitrate_50_half], dtype=np.float32)


def evaluate(answer_fname='result.csv'):
    predictions = {}
    with open('test_submit.csv', 'r') as fin:
        for line in fin:
            line = line.strip()
            if line == '':
                continue
            line = line.split(',')
            user_id = int(line[0])
            item_ids = [int(float(i)) for i in line[1:]]
            predictions[user_id] = item_ids

    answers = {}
    with open(answer_fname, 'r') as fin:
        for line in fin:
            line = [int(float(x)) for x in line.split(',')]
            user_id, item_id, item_degree = line
            answers[user_id] = (item_id, item_degree)

    scores = evaluate_each_file(predictions, answers)

    print('score is: ' + str(scores[0]*0.5+scores[1]*0.5))
    print('ndcg_50_full is: ' + str(scores[0]))
    print('ndcg_50_half is: ' + str(scores[1]))
    print('hitrate_50_full is: ' + str(scores[2]))
    print('hitrate_50_half is: ' + str(scores[3]))


evaluate()
