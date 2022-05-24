import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import metrics


##########################################################################
# functions that would be used in factor test ############################
##########################################################################

def level_separate(df, i):
    df_temp = df[df.rank_ == i].fillna(0).copy()
    df_temp = df_temp.groupby(['TRADE_DATE', 'industryName1'])[['ret', 'weight_ins']].mean().reset_index()
    sum_ = df_temp.groupby('TRADE_DATE').apply(lambda x: np.array(x.ret).dot(np.array(x.weight_ins)))
    return sum_


def cal_long_short_port(df, tier_num):
    first_ = (level_separate(df, 0) - level_separate(df, tier_num - 1) + 1).cumprod().shift(1)[-1]
    last_ = (level_separate(df, tier_num - 1) - level_separate(df, 0) + 1).cumprod().shift(1)[-1]
    if first_ > last_:
        long_short_port = level_separate(df, 0) - level_separate(df, tier_num - 1)
    else:
        long_short_port = level_separate(df, tier_num - 1) - level_separate(df, 0)
    long_short_port_cum = (long_short_port + 1).cumprod().shift(1)
    long_short_port_cum[0] = 1

    return long_short_port_cum, long_short_port


def plot_roc_auc(y_test, y_pred):
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = metrics.roc_auc_score(y_test, y_pred)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(" rise VS drop")
    plt.legend(loc="lower right")
    return auc  # need to be used in test all period


def plot_portfolio_cut(df, index_cum, tier_num):
    print('='*10 + 'df' + '='*10)
    print(df)
    plt.plot(index_cum, label='300_Index', color='red', linestyle=':')
    df_tier_ret = pd.DataFrame()
    df_tier_ret['index300'] = index_cum['closeIndex']
    for i in range(tier_num):
        tier_ret = level_separate(df, i)
        print('='*10 + 'tier_ret' + '='*10)
        print(tier_ret)
        df_tier_ret['tier' + str(i + 1)] = tier_ret
        plt.plot((tier_ret + 1).cumprod(), label='tier ' + str(i + 1))
        plt.legend()
    plt.title('portfolio cut by rank_')

    return df_tier_ret  # need to be used in test all period, would be concat by rows in main func


def plot_excess_return(df, index_cum, tier_num):
    for i in range(tier_num):
        t = ((level_separate(df, i) + 1).cumprod() - index_cum.T).ffill().T
        plt.plot(t, label='tier ' + str(i + 1))
        plt.legend()
    plt.title('excess return cut by rank_')


def plot_long_short(df, tier_num):
    long_short_port_cum, _ = cal_long_short_port(df, tier_num)  # we dont need long short port here

    plt.plot(long_short_port_cum)
    plt.title('long short portfolio performance')


def plot_all_evaluation(y_test, y_pred, df, index_cum, tier_num, period_idx, method,period_len, test_len,param_grid):
    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    auc = plot_roc_auc(y_test, y_pred)

    plt.subplot(2, 2, 2)
    df_tier_ret = plot_portfolio_cut(df, index_cum, tier_num)  # need to be used in test all period

    plt.subplot(2, 2, 3)
    plot_excess_return(df, index_cum, tier_num)

    plt.subplot(2, 2, 4)
    plot_long_short(df, tier_num)

    str_para = '_'.join([key+str(item) for key,item in param_grid.items()]) if param_grid else 'None'
    path = 'results/' + str(method) + '/' + str(period_len) + '_' + str(test_len) + '_' + str_para

    # Check whether the specified path exists or not
    isExist = os.path.exists(path)

    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    img_path = path + '/' + str(period_idx) + '.png'
    plt.savefig(img_path)
    # plt.show()

    return df_tier_ret, auc


##########################################################################
# print functions, just for printing something ... #######################
##########################################################################

# For printing the evaluation >>> IC, RIC , etc. Would be use in factor test
def print_evaluation(eval_list, eval_name):
    mean = eval_list.mean()
    std = eval_list.std()
    print('{}_mean = {}'.format(eval_name, mean))
    print('{}_std = {}'.format(eval_name, std))
    print('{}_positive_prob = {}'.format(eval_name, len(eval_list[eval_list >= 0]) / len(eval_list)))
    print('{}_IR = {}'.format(eval_name, mean / std))


def print_long_short(df, tier_num, num, is_period=True):
    if is_period:
        long_short_port_cum, long_short_port = cal_long_short_port(df, tier_num)
    else:
        long_short_port_cum, long_short_port = cal_long_short_all(df, tier_num)

    annual_ret = long_short_port_cum[-1] ** (1 / (len(long_short_port) / num)) - 1
    print('Annualized return of long-short portfolio: ' + str(format(annual_ret, '0.0%')))
    annual_vol = long_short_port.std() * (num ** 0.5)
    print('Sharp ratio of long-short portfolio: ' + str(round(annual_ret / annual_vol, 3)))
    print('Maximum draw down of long-short portfolio: ' +
          str(format(((long_short_port_cum - long_short_port_cum.cummax()) / long_short_port_cum.cummax()).min(),
                     '0.0%')))
    print('Win rate of long-short portfolio: ' + str(
        format(len(long_short_port[long_short_port > 0]) / len(long_short_port), '0.0%')))

##########################################################################
# functions that would be used in test all period ########################
##########################################################################


def cal_long_short_all(tier_ret, tier_num):
    tier_first = (tier_ret.iloc[:, 1] + 1).cumprod()[-1]
    tier_last = (tier_ret.iloc[:, tier_num] + 1).cumprod()[-1]
    if tier_first > tier_last:
        long_short_port = tier_ret.iloc[:, 1] - tier_ret.iloc[:, tier_num]
    else:
        long_short_port = tier_ret.iloc[:, tier_num] - tier_ret.iloc[:, 1]
    long_short_port_cum = (long_short_port+1).cumprod().shift(1)

    return long_short_port_cum, long_short_port


def plot_portfolio_cut_all(index_cum, tier_ret, tier_num):
    plt.plot(index_cum, label='300_Index', color='red', linestyle=':')
    # tier_ret['index300'] = index_cum
    for i in range(1, tier_num+1):
        plt.plot((tier_ret.iloc[:, i]+1).cumprod(), label='tier '+ str(i))
        plt.legend()
    plt.title('portfolio cut by rank_')


def plot_excess_return_all(index_cum, tier_ret, tier_num):
    for i in range(1, tier_num+1):
        plt.plot(((tier_ret.iloc[:, i]+1).cumprod()-index_cum.T).ffill().T, label='tier '+str(i))
        plt.legend()
    plt.title('excess return cut by rank_')


def plot_long_short_all(tier_ret, tier_num):
    long_short_port_cum, _ = cal_long_short_all(tier_ret, tier_num)
    plt.plot(long_short_port_cum)
    plt.title('long short portfolio performance')


def plot_roc_auc_all(auc_all):
    plt.plot(auc_all, label='auc')
    plt.legend()
    plt.title('AUC score')



def plot_test_all_period(index_cum, tier_ret, tier_num, auc_all, method,period_len, test_len,param_grid):
    plt.figure(figsize=(20, 20))

    plt.subplot(2, 2, 1)
    plot_portfolio_cut_all(index_cum, tier_ret, tier_num)

    plt.subplot(2, 2, 2)
    plot_excess_return_all(index_cum, tier_ret, tier_num)  # need to be used in test all period

    plt.subplot(2, 2, 3)
    plot_long_short_all(tier_ret, tier_num)

    plt.subplot(2, 2, 4)
    plot_roc_auc_all(auc_all)

    path = 'results/' + str(method)

    str_para = '_'.join([key+str(item) for key,item in param_grid.items()]) if param_grid else 'None'
    img_path = path + '/' + str(period_len) + '_' + str(test_len) + '_' + str_para + '/' + 'all.png'
    plt.savefig(img_path)
    # plt.show()

    auc_path = path + '/' + str(method) + '_auc_all.csv'
    tier_path = path + '/' + str(method) + '_ret_all.csv'
    auc_all = pd.DataFrame(auc_all,columns=['auc']).assign(period_len=period_len,test_len=test_len,param_grid=str_para)
    tier_ret = tier_ret.assign(period_len=period_len,test_len=test_len,param_grid=str_para)
    if not os.path.exists(auc_path):
        auc_all.to_csv(auc_path)
        tier_ret.to_csv(tier_path)
    else:
        auc_t = pd.read_csv(auc_path,index_col=0);tier_t = pd.read_csv(tier_path,index_col=0)
        auc_t = pd.concat([auc_t,auc_all],axis=0);tier_t = pd.concat([tier_t,tier_ret],axis=0)
        auc_t.to_csv(auc_path)
        tier_t.to_csv(tier_path)




