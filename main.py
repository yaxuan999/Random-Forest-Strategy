import pandas as pd

from utils import *
from visualization import *
root_path=  "D:\Sun Yaxuan\Msc\course\course work\8017\project\Data_Mining_Project-main\Data_Mining_Project-main"

# read data frame and preprocess the dataframe

df_factor_1 = pd.read_csv(root_path+'\df_factor_1.csv', index_col=0)
df_factor = pd.read_csv(root_path+'\df_factor.csv', index_col=0)
weight_ins = pd.read_csv(root_path+'\weight_ins.csv', index_col=0, parse_dates=True)
Industry = pd.read_csv(root_path+'\Industry18-22.csv', index_col=0)

index_cum = pd.read_csv(root_path+'\index_cum.csv', parse_dates=True)  # we need TRADE DATE columns for later comparison

df_factor_1 = preprocessing(df_factor_1)
df_factor = preprocessing(df_factor)
weight_ins = preprocessing(weight_ins)
Industry = preprocessing(Industry)
index_cum = preprocessing(index_cum)

# get all the date list
date_list = df_factor_1.TRADE_DATE.unique()


# config
def period_training(start_train, end_train, end_test, param_grid=None, methods=['rf'],k=5):

    train_X, train_y, X_test, y_test = dataloader(start_train, end_train, end_test, df=df_factor_1)

    # predict
    # param_grid = {'max_leaf_nodes': [30], 'min_samples_leaf': [5], 'max_features': [8]}
    pred_rise = training(train_X, X_test, train_y, k, param_grid=param_grid, methods=methods)
    test_period = X_test['TRADE_DATE'].unique()

    # when merging tables, we mainly use these 2 columns together to locate the row
    idx_col = ['TRADE_DATE', 'TICKER_SYMBOL']
    df_test_ret = get_ret(df_factor)

    # we only need the data those match the time of validation set
    df_test_ret = df_test_ret[df_test_ret['TRADE_DATE'].isin(test_period)]
    df_Industry = Industry[Industry['TRADE_DATE'].isin(test_period)]
    df_weight_ins = weight_ins[weight_ins['TRADE_DATE'].isin(test_period)]

    # merge predict rise with other three tables on index columns
    pred_rise = pd.merge(pred_rise, df_test_ret, on=idx_col, how='left').dropna(subset=['ret'])
    pred_rise = pd.merge(pred_rise, df_Industry, on=idx_col, how='left').dropna(how='any')
    pred_rise = pd.merge(pred_rise, df_weight_ins, on=['TRADE_DATE', 'industryName1'], how='left').dropna(how='any')

    # pred_rise = pred_rise.drop_duplicates(keep='first')
    # print('='*20)
    df_test_estimator_rank = stock_rank(df=pred_rise)
    df_test_estimator_rank = preprocessing(df_test_estimator_rank)  # drop the duplicated rows, transform datetime format
    df_test_estimator_rank = pd.merge(df_test_estimator_rank, pred_rise, on=idx_col, how='left')

    return df_test_estimator_rank


def main_func(period_len, test_len, param_grid=None, methods=['lr'],k=5):
    num_epoch = (len(date_list)-period_len)//test_len
    df_tier_ret = pd.DataFrame()  # use to record all the evaluation and output from factor test
    auc_all = []
    method = '_'.join(methods)
    for i in range(0, num_epoch):
        print('#'*15 + ' epoch {} '.format(i) + '#'*15)

        idx = i*test_len
        start_train = date_list[idx]
        end_train = date_list[idx + period_len - test_len]
        end_test = date_list[idx + period_len]

        df_val_estimator_rank = period_training(start_train, end_train, end_test, param_grid=param_grid, methods=methods,k=k)

        # plot_roc_auc(y_test, y_pred)
        mask_index_cum = (index_cum['TRADE_DATE'] >= end_train) & (index_cum['TRADE_DATE'] < end_test)
        index_cum_period = index_cum.loc[mask_index_cum].copy()
        index_cum_period.set_index('TRADE_DATE', inplace=True)
        
        df_tier_ret_tmp, auc = factor_test(df_val_estimator_rank, index_cum=index_cum_period,
                                           period_idx=i, method=method, period_len=period_len, test_len=test_len,param_grid=param_grid, freq='BM', tier_num=5)
        df_tier_ret = df_tier_ret.append(df_tier_ret_tmp)
        auc_all.append(auc)

    # ret == 0 need to be removed
    df_tier_ret.reset_index(inplace=True)
    df_tier_ret = df_tier_ret.rename(columns={'index': 'TRADE_DATE'})
    df_tier_ret = preprocessing(df_tier_ret)  # actually means nothing, just make sure pd datetime format

    start = df_tier_ret['TRADE_DATE'].min()
    end = df_tier_ret['TRADE_DATE'].max()
    df_tier_ret.set_index('TRADE_DATE', inplace=True)  # set it back to prevent error

    mask_index_cum_all = (index_cum['TRADE_DATE'] >= start) & (index_cum['TRADE_DATE'] <= end)
    index_cum_all = index_cum.loc[mask_index_cum_all].copy()
    index_cum_all.set_index('TRADE_DATE', inplace=True)

    test_all_period(index_cum_all, df_tier_ret, auc_all,method, period_len, test_len, param_grid, freq='BM', tier_num=5)


if __name__ == '__main__':
    main_func(period_len=24, test_len=6,methods=['ab','lr','gb','rf'],k=1)
