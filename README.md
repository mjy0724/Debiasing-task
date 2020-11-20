Debiasing
===================================
This is a simple implementation of an item-based collaborative filtering algorithm and does not make use of features to improve accuracy. You can optimize on this version, or use a completely different method for this project.

## Files

1. `baseline.py`: An example of code submission you can use as template. You can also write your own code without referring to this baseline.
2. `evaluation.py`: Please use this file to evaluate your code and obtain the final score.
3. `result.csv`: This is the correct result for `test_id.csv`. This file is used to calculate the final score.


## Test

1. Run baseline.py and output the 51-column prediction to a csv file.
2. Run evaluation.py to compare the predicted results with the real results and calculate the final score.

## Dataset (uploaded to WEB LEARNING before)

### The training data: 

1. `underexpose_item_feat.csv`: the columns of which are: *item_id, txt_vec, img_vec*
2. `underexpose_user_feat.csv`: the columns of which are: *user_id, user_age_level, user_gender, user_city_level*
3. `train.csv`: columns are: user_id, item_id, time

### The test data:

`test_id.csv`: *user_id, query_time*

### Column:

1. **txt_vec**：the item's text feature, which is a 128-dimensional real-valued vector produced by a pre-trained model
2. **img_vec**：the item's image feature, which is a 128-dimensional real-valued vector produced by a pre-trained model
3. **user_id**：the unique identifier of the user
4. **item_id**：the unique identifier of the item
5. **time**：timestamp when the click event happens, i.e.,（unix_timestamp - random_number_1）/ random_number_2
6. **user_age_level**：the age group to which the user belongs
7. **user_gender**：the gender of the user, which can be empty
8. **user_city_level**：the tier to which the user's city belongs


