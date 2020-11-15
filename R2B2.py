#Interpretable Next Basket Prediction Boosted with Representative Recipes - 2020
#Flavour Aware Food Rec System

#R2B2

#USAGE
#python R2B2.py Arguments1 Arguments2 Arguments3

#Arguments 1 - type_recc - Values: norecipes|20recipes|10recipes|5recipes|2recipes|1recipes
#Arguments 2 - distance - Values: jaccard|pearson|cosine
#Arguments 3 - alpha1 - Values: 0|0.1|0.2|0.3|0.4|0.5|0.6|0.7|0.8|0.9|1

#Example: python R2B2.py 20recipes cosine 0.3

#Import Libraries
import turicreate as tc
import pandas as pd
from turicreate.toolkits.recommender.util import precision_recall_by_user
import sys

#Read Transactional Dataset
dataset = pd.read_csv('dataset/data_tran_rating.csv', sep=";", header=0,verbose=False)

#Read Items Dataset
item_data=tc.SFrame.read_csv("dataset/data_product.csv", header=True, delimiter=';',verbose=False)
#Default Items Fields
item_data = item_data['product_id','aisle_id','department_id','quantity_sold','ratio_reordered']

#Read Users Dataset
user_data=tc.SFrame.read_csv("dataset/data_user_wrecipes.csv", header=True, delimiter=';',verbose=False)

#Different Number of Recipes in the Users Dimension
type_recc=sys.argv[1]

if type_recc == 'norecipes':
	#No Recipes
	user_data = user_data['user_id', "max_quant_for_order",'number_order', 'avg_quant_for_order','ratio_reordered','stdev_quant_for_order']
elif type_recc == '20recipes':
	#20 Recipes
	user_data = user_data['user_id', "max_quant_for_order",'number_order', 'avg_quant_for_order','ratio_reordered','stdev_quant_for_order',"recipes_1","recipes_2","recipes_3","recipes_4","recipes_5","recipes_6","recipes_7","recipes_8","recipes_9","recipes_10","recipes_11","recipes_12","recipes_13","recipes_14","recipes_15","recipes_16","recipes_17","recipes_18","recipes_19","recipes_20"]
elif type_recc == '10recipes':
	#10 Recipes
	user_data = user_data['user_id', "max_quant_for_order",'number_order', 'avg_quant_for_order','ratio_reordered','stdev_quant_for_order',"recipes_1","recipes_2","recipes_3","recipes_4","recipes_5","recipes_6","recipes_7","recipes_8","recipes_9","recipes_10"]
elif type_recc == '5recipes':
	#5 Recipes
	user_data = user_data['user_id', "max_quant_for_order",'number_order', 'avg_quant_for_order','ratio_reordered','stdev_quant_for_order', "recipes_1","recipes_2","recipes_3","recipes_4","recipes_5"]
elif type_recc == '2recipes':
	#2Recipes
	user_data = user_data['user_id', "max_quant_for_order",'number_order', 'avg_quant_for_order','ratio_reordered','stdev_quant_for_order', "recipes_1","recipes_2"]
elif type_recc == '1recipes':
	#1Recipes
	user_data = user_data['user_id', "max_quant_for_order",'number_order', 'avg_quant_for_order','ratio_reordered','stdev_quant_for_order', "recipes_1"]
else:
	sys.exit()

# Model Develop and Test
# Recipes Weights
alpha1 = [float(sys.argv[3])]
for i in alpha1:
	alpha2 = 1 - i
	tmp = dataset
	tmp['newcolumn'] = (alpha1 * tmp['rating_norm']) + (alpha2 * tmp['rating_recipes'])
	data = tc.SFrame(data=tmp)
	data = data['user_id','product_id','newcolumn']

	#Create Training Set and Test Set
	train, test = tc.recommender.util.random_split_by_user(data, user_id='user_id', item_id='product_id',item_test_proportion=0.8)

	#Develop Model
	#Distance Selection

	distance=sys.argv[2]

	if distance == 'jaccard':
		#Jaccard Distance Model
		m_jacc = tc.item_similarity_recommender.create(train, user_id='user_id', item_id='product_id',target='newcolumn', similarity_type='jaccard', user_data =user_data, item_data=item_data, verbose=False)
		#Print Result
		print ''
		print '-------------------' + 'JACCARD ------' + 'alpha1: ' + str(i) + '-----' + 'alpha2: ' + str(alpha2) + '-----------------'
		print ''
		print m_jacc.evaluate_precision_recall(test, cutoffs=[1,5, 10,20,50,100],verbose=False)
	elif distance == 'pearson':
		#Pearson Distance Model
		m_pearson = tc.item_similarity_recommender.create(train, user_id='user_id', item_id='product_id',target='newcolumn', similarity_type='pearson', user_data =user_data, item_data=item_data, verbose=False)
		#Print Result
		print ''
		print '-------------------' + 'PEARSON ------' + 'alpha1: ' + str(i) + '-----' + 'alpha2: ' + str(alpha2) + '-----------------'
		print ''
		print m_pearson.evaluate_precision_recall(test, cutoffs=[1,5, 10,20,50,100],verbose=False)

	elif distance == 'cosine':
		#Cosine Distance Model
		m_cos = tc.item_similarity_recommender.create(train, user_id='user_id', item_id='product_id',target='newcolumn', similarity_type='cosine', user_data =user_data, item_data=item_data, verbose=False)
		#Print Result
		print ''
		print '-------------------' + 'COSINE ------' + 'alpha1: ' + str(i) + '-----' + 'alpha2: ' + str(alpha2) + '-----------------'
		print ''
		print m_cos.evaluate_precision_recall(test, cutoffs=[1,5, 10,20,50,100],verbose=False)
	else:
		sys.exit()