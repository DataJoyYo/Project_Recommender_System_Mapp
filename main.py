# Import libraries
import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
from pandas.api.types import CategoricalDtype
import warnings
warnings.filterwarnings("ignore")

# The weighting scheme that each action is going to be weighted as
mainActionMat = {'view': 2, 'add': 8, 'buy': 10, 'mainCat-price interest': 2,'subCat-price interest':6,'mainCat interest':4}

# Tuning parameters for the model
alphaVal=15
regulizationValue=0.4
interationsValue=5
factorsValue=200
# Threshold for cleaning inactive users and unpopular products
userCutover=1500
productCutover=500

# list of actions that model is going to be tested on
runs=['buy','add','view']

def product_price_bucket_sub(df, action_type_strength):
    '''Function takes the Training dataset and the action weighting.
    Function calculates the most popular product for each subcategory and price bucket

    example: the product xyz has the highest interest in Calvin Klein in price bucket 3 (22 unique buckets)

    Function returns the dataframe with unique products subcategory and price bucket
    '''
    df['priceBucket'] = pd.cut(df['PRICE'],22, labels = False)
    df['rating'] = df['action'].apply(lambda x: action_type_strength[x])
    df['product_total'] = df.groupby('product')['rating'].transform('sum')
    idx = df.groupby(['subCatName', 'priceBucket'])['product_total'].transform(max) == df['product_total']
    df = df[idx]
    df = df[['product', 'subCatName', 'priceBucket']]
    product_df = df.drop_duplicates()
    return product_df


def user_price_bucket_sub(df, action_type_strength):
    '''Function takes the Training dataset and the action weighting.
    Function calculates the most interesting subcategory and pricebucket for each user

    example: user xyz is mostly interested in Calvin Klein in price bucket 3 (22 price buckets)

    Function returns the dataframe with unique users, subcategory and price bucket
    '''

    df['priceBucket'] = pd.cut(df['PRICE'],22, labels = False)
    df['rating'] = df['action'].apply(lambda x: action_type_strength[x])
    df['product_total'] = df.groupby('user')['rating'].transform('sum')
    idx = df.groupby(['subCatName', 'priceBucket'])['product_total'].transform(max) == df['product_total']
    df = df[idx]
    df = df[['user', 'subCatName', 'priceBucket']]
    user_df = df.drop_duplicates()
    return user_df

def product_price_bucket_main(df, action_type_strength):
    '''Function takes the Training dataset and the action weighting.
       Function calculates the most popular product for each mainCategory and price bucket

       example: the product xyz has the highest interest in parfume in price bucket 3 (22 unique buckets)

       Function returns the dataframe with unique products MainCategory and price bucket
    '''
    df['priceBucket'] = pd.cut(df['PRICE'],22, labels = False)
    df['rating'] = df['action'].apply(lambda x: action_type_strength[x])
    df['product_total'] = df.groupby('product')['rating'].transform('sum')
    idx = df.groupby(['mainCatName', 'priceBucket'])['product_total'].transform(max) == df['product_total']
    df = df[idx]
    df = df[['product', 'mainCatName', 'priceBucket']]
    product_df = df.drop_duplicates()
    return product_df


def user_price_bucket_main(df, action_type_strength):
    '''Function takes the Training dataset and the action weighting.
     Function calculates the most interesting MainCategory and pricebucket for each user

     example: user xyz is mostly interested in parfume in price bucket 3 (22 price buckets)

     Function returns the dataframe with unique users, MainCategory and price bucket
     '''
    df['priceBucket'] = pd.cut(df['PRICE'],22, labels = False)

    df['rating'] = df['action'].apply(lambda x: action_type_strength[x])
    df['product_total'] = df.groupby('user')['rating'].transform('sum')
    idx = df.groupby(['mainCatName', 'priceBucket'])['product_total'].transform(max) == df['product_total']
    df = df[idx]
    df = df[['user', 'mainCatName', 'priceBucket']]
    user_df = df.drop_duplicates()
    return user_df


def product_bucket_main(df, action_type_strength):
    '''Function takes the Training dataset and the action weighting.
       Function calculates the most popular product for each mainCategory

       example: the product xyz has the highest interest in parfume

       Function returns the dataframe with unique products mainCategory
       '''
    df['rating'] = df['action'].apply(lambda x: action_type_strength[x])
    df['product_total'] = df.groupby('product')['rating'].transform('sum')
    idx = df.groupby(['mainCatName'])['product_total'].transform(max) == df['product_total']
    df = df[idx]
    df = df[['product', 'mainCatName']]
    product_df = df.drop_duplicates()
    return product_df


def user_bucket_main(df, action_type_strength):
    '''Function takes the Training dataset and the action weighting
         Function calculates the most interesting MainCategory each user

         example: user xyz is mostly interested in parfume

         Function returns the dataframe with unique users, MainCategory
         '''
    df['rating'] = df['action'].apply(lambda x: action_type_strength[x])
    df['product_total'] = df.groupby('user')['rating'].transform('sum')
    idx = df.groupby(['mainCatName'])['product_total'].transform(max) == df['product_total']
    df = df[idx]
    df = df[['user', 'mainCatName']]
    user_df = df.drop_duplicates()
    return user_df


def createDataFrame(df, action_type_strength,run):
    '''df: Raw dataset of cleaned data
    Adds the rating based on the dictionary action_type_strength
    RETURNS Dataframe with rated interactions
    '''

    '''Creating new actions from user_bucket_main, user_price_bucket_main, user_price_bucket_sub and create new relationships
    if such relationship exists, the total weight of the relationship will be increased
    '''
     
    product_pop = product_price_bucket_main(df, action_type_strength)
    user_pop = user_price_bucket_main(df, action_type_strength)
    new_mat_main = user_pop.merge(product_pop, how='left', on=['mainCatName', 'priceBucket'])
    new_mat_main['action'] = 'mainCat-price interest'

    product_pop = product_price_bucket_sub(df, action_type_strength)
    user_pop = user_price_bucket_sub(df, action_type_strength)
    new_mat_sub = user_pop.merge(product_pop, how='left', on=['subCatName', 'priceBucket'])
    new_mat_sub['action'] = 'subCat-price interest'

    product_pop = product_price_bucket_main(df, action_type_strength)
    user_pop = user_price_bucket_main(df, action_type_strength)
    new_mat_main_only = user_pop.merge(product_pop, how='left', on=['mainCatName', 'priceBucket'])
    new_mat_main_only['action'] = 'mainCat interest'

    df = pd.concat([new_mat_main, df,new_mat_main_only,new_mat_sub])
    df1 = df[['user', 'product', 'action']]
    
    # Adding all the weights together to create unique user - price- total rating dataframe
    #df1 is the main traning ready dataframe that will be used to train our model
    df1['rating'] = df1['action'].transform(lambda x: action_type_strength[x])
    df1 = df1[['user', 'product', 'rating']]
    df1 = df1.groupby(['user', 'product']).sum().reset_index()
    df1 = df1.drop_duplicates()
    
    #df2 is the dataframe with 99999 weight. For the user-product interactions that already exists for each run, approrpiate action will be wieghted as the 99999.
    #this dataframe will be used to make sure we do not recommend any items that have already been interacted.
    buy=0
    add=0
    view=0
    if (run=='buy'):
        buy=99999
    elif(run=='add'):
        buy = 99999
        add = 99999
    else:
        buy = 99999
        add = 99999
        view=99999
    
    df2 = df[['user', 'product', 'action']]
    action_type_strength = {'view': view, 'add': add, 'buy': buy, 'mainCat-price interest': 0,'subCat-price interest':0,'mainCat interest':0}
    df2['rating'] = df2['action'].apply(lambda x: action_type_strength[x])
    df2 = df2[['user', 'product', 'rating']]
    df2 = df2.groupby(['user', 'product']).sum().reset_index()
    df2 = df2.drop_duplicates()
    
    return df1, df2


def split_train_test(df):
    '''Function splits the total dataset into training set(before April) and testing set (April).
    Function also creates two different lists of data:
    df_test - all the interactions having users that have been seen in df_train
    df_missing - all the interactions having users that have NOT been seein in df_train

    '''
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df_train = df[df['month'] < 4]
    df_test = df[df['month'] == 4]
    val1 = df_test.shape[0]
    print('Number of Rows in raw test: ' + str(val1))
    
    #remove the users that did NOT appear in the training set out of df_test and save them in df_mising.
    df_missing=df_test[~df_test.user.isin(df_train.user)]
    df_test = df_test[df_test.user.isin(df_train.user)]
    print('Number of Rows after cleaning in test: ' + str(df_test.shape[0]))
    print('Number of Rows removed from test: ' + str(val1 - df_test.shape[0]))
    df_test = df_test.drop(['month'], axis=1)
    df_train = df_train.drop(['month'], axis=1)
    return df_train, df_test,df_missing


def cleanInActive(df_train, df_test, action_type_strength,userCu,productCu):
    '''Function: Based on the cutover point of users and cutover point of products,
        function will remove all the products and users less than the cutover point from df_train except users appear in df_test.
        Because we do not want to remove any information for users that we need to predict for
    '''
    df_train['rating'] = df_train['action'].apply(lambda x: action_type_strength[x])
    df_train['user_total'] = df_train.groupby('user')['rating'].transform('sum')
    df_train['product_total'] = df_train.groupby('product')['rating'].transform('sum')
    df_train = df_train[df_train['user'].isin(df_test['user']) | (df_train['user_total'] >= userCu)]
    df_train = df_train[df_train['user'].isin(df_test['user']) | (df_train['product_total'] >= productCu)]
    return df_train, df_test

# reading the dataframe
df = pd.read_csv('/home/schaal/analytics2_train.csv')

# for each run: buy, view, add
for run in runs:
    # split data into train, test, and df_missing
    df_train, df_test,df_ul = split_train_test(df)
    #clean inactive users and products
    df_train, df_test = cleanInActive(df_train, df_test, mainActionMat,userCutover,productCutover)
    # create the model ready dataframe and give dataframe with existing interactions for the run
    df_action_mat, df_excisting_interactions = createDataFrame(df_train, mainActionMat,run)
    # cleaning testing set for the respective run
    if (run == 'buy'):
        df_test = df_test[df_test['action'] == 'buy']
        df_ul = df_ul[df_ul['action'] == 'buy']

    elif (run == 'add'):
        df_test = df_test[df_test['action'] != 'view']
        df_ul = df_ul[df_ul['action'] != 'view']
    # getting testing set ready for hit rate calculation.
    df_test = df_test.groupby(['user', 'product']).count().reset_index()
    df_test = df_test[['user', 'product']]
    df_test['user_product'] = df_test['user'] + '/' + df_test['product']
    testList = df_test['user'].unique().tolist()
    # creating 3 implicit rating matrixes
    # df_train - user_product , product_user (used for modeling)
    #df_excisting_interactions - user_product with either 0 or >=99999
    users = list(np.sort(df_action_mat.user.unique()))  # users
    products = list(df_action_mat['product'].unique())  # products
    ratings = list(df_action_mat.rating)
    print(len(ratings))
    print(len(users))
    print(len(products))

    buy_rating = list(df_excisting_interactions.rating)

    cols = df_action_mat.user.astype(CategoricalDtype(categories=users)).cat.codes
    rows = df_action_mat['product'].astype(CategoricalDtype(categories=products)).cat.codes
    sparse_item_user = sparse.csr_matrix((ratings, (rows, cols)), shape=(len(products), len(users)))

    rows = df_action_mat.user.astype(CategoricalDtype(categories=users)).cat.codes
    cols = df_action_mat['product'].astype(CategoricalDtype(categories=products)).cat.codes
    sparse_user_item = sparse.csr_matrix((ratings, (rows, cols)), shape=(len(users), len(products)))

    rows = df_excisting_interactions.user.astype(CategoricalDtype(categories=users)).cat.codes
    cols = df_excisting_interactions['product'].astype(CategoricalDtype(categories=products)).cat.codes
    buy_user_item = sparse.csr_matrix((ratings, (rows, cols)), shape=(len(users), len(products)))


    # create model with parameters
    model = implicit.als.AlternatingLeastSquares(factors=factorsValue, regularization=regulizationValue, iterations=interationsValue)
    alpha_val = alphaVal
    # multiply by alphaVal to make sure we do not overfit
    data_conf = (sparse_item_user * alpha_val).astype('double')
    # fit the model to the train set
    model.fit(data_conf)
    # recommend 15 items for each user
    myList = (model.recommend_all(sparse_user_item, N=15, filter_already_liked_items=False))
    # create the user and product np arrays that will match with idx of our sparese matrix
    customers_arr = np.array(users)  # Array of customer IDs from the ratings matrix
    products_arr = np.array(products)

    recommend_item = []
    recommend_user = []
    from tqdm import tqdm
    # iterate for each user in our df_test
    for testUser in tqdm(testList):
        finalReccomendations = []
        if (testUser not in customers_arr):
            print('missing')
            continue
        # get the index of the user in our implicit rating matrix
        user_ind = np.where(customers_arr == testUser)[0][0]
        prf_vector = myList[user_ind, :]
        # from the 15 recommended items, pick top 3 items that have rating < 99998 in df_excisting_interactions
        for recs in prf_vector:
            if (buy_user_item[user_ind, recs] > 99998):
                continue
            else:
                finalReccomendations.append(recs)
        # create lists of users and top 3 items for each user.
        for number in range(0, 3):
            item_index = finalReccomendations[number]
            recommend_user.append(testUser)
            recommend_item.append(products_arr[item_index])
            
    # prepare the reccomendation dataframe
    dct = {'test_user': recommend_user, 'test_product': recommend_item}
    df_rec = pd.DataFrame(dct)
    df_rec['user_product'] = df_rec['test_user'].astype(str) + '/' + df_rec['test_product'].astype(str)
    
    # merge df_test with reccomendation dataframe
    df_merged = df_test.merge(df_rec, on='user_product', how='inner')
    
    # calculate hit rate
    numTestUsers=len(testList)
    numTestIteractions=df_test.shape[0]
    numReccomendations=df_rec.shape[0]
    numHits=df_merged.shape[0]
    percentHits=numHits/numReccomendations

    totalWUL=len(df_ul['user'].unique())*3+numReccomendations
    percentHitswUL=numHits/totalWUL

    # print results for each run
    #hit rate is calculated in two ways:
    #1)only based on users in df_test that appear in df_train
    #2) to be consistent with group3: based on all users in df_test, not matter users are in df_train or not 
    #meaning that no matter the model has the chance to learn the behaviours in df_train or not.
    print ('Run for: '+run)
    print('Number of Users in Test: ' + str(numTestUsers))
    print('Number of Interactions in Test:' + str(numTestIteractions))
    print('Number of Recommendations: ' + str(numReccomendations))
    print('Number of Hits: ' + str(numHits))
    print('% of Hits for users that only appeared in df_train: ' + str(percentHits))
    print ('INCLUDING USERS NOT Appearing in df_train')
    print ('NUMBER OF RECCOMENDATIONS including Users NOT Appearing in df_train: '+str(totalWUL))
    print('% of Hits for total users (including users Not Appearing in df_train): ' + str(percentHitswUL))


