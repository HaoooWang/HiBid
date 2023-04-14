# -*- coding: utf-8 -*-
'''
High-level status : Within the past H days, the number of PVs obtained for each I-th channel per day, the daily transaction volume, cost, number of transactions, clicks, successful bids, total bids, daily success rate, click-through rate, conversion rate, budget, target ROI reference value, and real ROI per I-th channel, the total number of PVs for all channels per day, the total transaction volume, cost, number of transactions, clicks, successful bids, total bids, comprehensive success rate, click-through rate, conversion rate, total budget, total target ROI, and total real ROI for all channels per day, and the budget and ROI of the advertise for that day.
High-level action : Allocate the budget reference value weight, allocate the GMV reference value weight (calculate to obtain the budget ROI reference value).
High-level reward : The comprehensive click-through rate for the advertise on all channels.

Low-level status information: advertise information in this channel: advertise ID, the current day's PV number, current bidding day, current channel, the number of PVs for the current channel, user type, user preference distribution for the type of advertise, user preference distribution for the channel, advertise type, advertise average CTR/CVR/GMV, the current day's transaction volume, cost, number of transactions, clicks, successful bids, total bids, daily success rate, click-through rate, conversion rate, budget reference value, GMV reference value, ROI reference value, remaining amount relative to budget reference value, and achieved ROI.
Information of other channels: The number of PVs obtained for the I-th channel, the daily transaction volume, cost, number of transactions, clicks, successful bids, total bids, daily success rate, click-through rate, conversion rate, budget reference value, GMV reference value, ROI reference value, remaining amount relative to budget reference value, and achieved ROI for the advertise on the I-th channel.
Comprehensive advertise information: The total transaction volume, cost, number of transactions, clicks, successful bids, total bids, comprehensive success rate, click-through rate, conversion rate, total budget, total ROI, remaining budget of the advertise, and current achieved ROI for the advertise on all channels.
'''
import argparse
import numpy as np
import random

parser = argparse.ArgumentParser(description="simulation")
parser.add_argument('--Number_advertisers',default=5000,type=int)#系统中商家的个数
parser.add_argument('--Number_days',default=7,type=int)#正式统计的天数
# parser.add_argument('--High_cycle_days',default=3,type=int)#高层期天数
parser.add_argument('--Number_pvs',default=500000,type=int)#每天的PV个数
parser.add_argument('--Number_type_of_store',default=20,type=int)#商家类型个数
parser.add_argument('--Number_of_channels',default=4,type=int)#渠道个数
parser.add_argument('--Number_advertisers_for_pv',default=20,type=int)#每次竞价参与的最大商家数量
# parser.add_argument('--delta',default=0.01,type=float)#计算奖励时的参数
parser.add_argument('--Max_budget',default=1000,type=int)
parser.add_argument('--Min_budget',default=0,type=int)
parser.add_argument('--Max_bid',default=1,type=int)
parser.add_argument('--Min_bid',default=0,type=int)
parser.add_argument('--Max_ROI',default=20,type=int)
parser.add_argument('--Min_ROI',default=0,type=int)
parser.add_argument('--Max_GMV',default=50,type=int)


def main(args):
    Number_advertisers = args.Number_advertisers  # 商家数量
    Number_days = args.Number_days  # 天数
    # High_cycle_days = args.High_cycle_days #高层周期天数
    Number_pvs = args.Number_pvs  # pv流量数
    Number_type_of_store = args.Number_type_of_store  # 商家类型数
    Number_of_channels = args.Number_of_channels  # 渠道数
    Number_advertisers_for_pv = args.Number_advertisers_for_pv  # pv商家数：每个pv流量下参与竞价的商家的数量
    # delta = args.delta    # 计算奖励时的参数
    Max_budget = args.Max_budget
    Min_budget = args.Min_budget
    Max_bid = args.Max_bid
    Min_bid = args.Min_bid
    Max_ROI = args.Max_ROI
    Min_ROI = args.Min_ROI
    Max_GMV = args.Max_GMV
    store_CTR = np.zeros(Number_advertisers)  # 初始化，每个商家的平均点击率
    store_CPC = np.zeros(Number_advertisers)  # 初始化，每个商家的平均点击率
    store_CVR = np.zeros(Number_advertisers)  # 初始化，每个商家的平均成交率
    store_GMV = np.zeros(Number_advertisers)  # 初始化，每个商家的平均成交额
    store_type = np.zeros(Number_advertisers)  # 初始化，每个商家的类型
    # 最终的数据储存
    # saved_day = []
    saved_state_h = []
    saved_action_b_h = []
    saved_action_g_h = []
    saved_reward_h = []
    # saved_cost_h = []
    # saved_gmv_h = []
    # saved_ROI_h = []
    saved_next_state_h = []
    saved_terminal_h = []
    saved_state_l = [[] for i in range(Number_of_channels)]
    saved_action_l = [[] for i in range(Number_of_channels)]
    saved_reward_l = [[] for i in range(Number_of_channels)]
    saved_cost_l = [[] for i in range(Number_of_channels)]
    saved_gmv_l = [[] for i in range(Number_of_channels)]
    saved_ROI_l = [[] for i in range(Number_of_channels)]
    saved_next_state_l = [[] for i in range(Number_of_channels)]
    saved_terminal_l = [[] for i in range(Number_of_channels)]
    # all_hist_gmv = np.zeros(Number_days, Number_advertisers)  # 初始化，所有历史的成交额（每天每个商家）
    # 对于每个渠道来说，每个商家每天在该渠道的成交额，成本，成交数，点击数，竞价成功数，竞价总次数，竞价成功率，点击率，成交率，预算，ROI，真实ROI
    all_hist_gmv_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_cost_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_gmvn_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_clickn_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_winn_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_bidn_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_winr_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_ctr_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_cvr_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_bud_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_roi_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_real_roi_in_chan = np.zeros([Number_days, Number_advertisers, Number_of_channels])
    all_hist_chan_pv = np.zeros([Number_days, Number_of_channels])    #每天各渠道总pv
    all_hist_roi = np.zeros([Number_days, Number_advertisers])    #每天每个商家的ROI
    attend_bid_time = np.zeros(Number_advertisers)  # 初始化，各商家竞价成功次数？？？
    for store in range(Number_advertisers):  # 每个商家的平均点击率，平均成交率，平均成交额，商家类型
        store_CTR[store] = random.uniform(-1, 1)*0.01+0.027
        store_CVR[store] = random.uniform(-1, 1)*0.1+0.17
        store_CPC[store] = random.uniform(0.7, 1.4)
        store_GMV[store] = random.uniform(-10, 10)+40    # 单次成交数最好是多大
        store_type[store] = np.random.randint(Number_type_of_store)

    for day in range(Number_days):    # 先验数据生成
        # 初始化当天所有商家的历史效益
        # 初始化，当天每个商家在每个渠道上的成交额，成本，成交数，点击数，竞价成功数，参与竞价次数
        today_gmv_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_cost_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_gmvn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_clickn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_winn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_bidn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_chan_pv = np.zeros(Number_of_channels)  # 当天各渠道总pv
        # 每个商家为每个渠道分配当天的预算和ROI
        store_budget = np.zeros(Number_advertisers)    # 初始化，每个商家当天的预算
        store_budget_leave = np.zeros(Number_advertisers)    # 初始化，商家竞价时的剩余预算，最初等于商家当天的预算
        store_ROI = np.zeros(Number_advertisers)    # 初始化，每个商家当天的ROI
        store_budget_in_chan = np.zeros([Number_advertisers, Number_of_channels])    # 初始化，每个商家在每个渠道当天的预算参考值
        store_gmv_in_chan = np.zeros([Number_advertisers, Number_of_channels])  # 初始化，每个商家在每个渠道当天的GMV参考值
        store_ROI_in_chan = np.zeros([Number_advertisers, Number_of_channels])    # 初始化，每个商家在每个渠道当天的ROI参考值
        for store in range(Number_advertisers):
            store_budget[store] = random.uniform(Min_budget, Max_budget)    # 预算值多少比较好
            store_ROI[store] = random.uniform(Min_ROI, Max_ROI)    # ROI值多少比较好
            budget_weight = np.random.random(Number_of_channels)    # 给各渠道的预算参考值权重
            flag = True    #非零处理
            while flag:
                flag = False
                for c in range(Number_of_channels):
                    if budget_weight[c] == 0.0:
                        flag = True
                        budget_weight[c] = random.uniform(0, 1)
            store_budget_in_chan[store] = (budget_weight / budget_weight.sum()) * store_budget[store]
            gmv_weight = np.random.random(Number_of_channels)  # 给各渠道的GMV参考值权重
            store_gmv_in_chan[store] = (gmv_weight / gmv_weight.sum()) * store_budget[store] * store_ROI[store]
            # store_ROI_in_chan[store] = [store_gmv_in_chan[store][chan] / store_budget_in_chan[store][chan] for chan in range(Number_of_channels)]    # 给各渠道的ROI参考值
            store_ROI_in_chan[store] = store_gmv_in_chan[store] / store_budget_in_chan[store]  # 给各渠道的ROI参考值
        store_budget_leave = store_budget * 1

        for pv in range(Number_pvs):    #遍历当天每一条pv流量
            # print('pre_pv: ', pv)
            print('pre_day: ', day, '......', 'pre_pv: ', pv)
            user_type = np.zeros(3)    #初始化，当前用户在点击率，转化率，成交额方面的类型
            bool_click = 0  # 是否点击
            bool_convert = 0  # 是否转化
            store_prefer = np.zeros(Number_type_of_store)  # 初始化，维度为商家类型数量的先验，代表当前用户对各类型商家的喜好程度（喜欢烧烤、面食等情况）
            channel_prefer = np.zeros(Number_of_channels)  # 初始化，维度为渠道数量的先验，代表当前用户对各渠道的喜爱程度
            channel_sum = np.zeros(Number_of_channels)  # 初始化，维度为渠道数量，代表[渠道一喜爱程度,渠道一加二喜爱程度,渠道一加二加三喜爱程度,...]
            stores_for_pv = np.zeros(Number_advertisers_for_pv)  # 初始化，维度为pv商家数，代表参与当前pv流量竞价的商家
            store_bid = np.zeros(Number_advertisers_for_pv)  # 初始化，维度为pv商家数，代表参与竞价的商家出的价
            # random_value1 = np.random.rand()
            # random_value2 = np.random.rand()
            # random_value3 = np.random.rand()
            # # 决定用户点击率
            # if random_value1 < 1.0 / 3.0:
            #     user_ctr = 0.8
            #     user_type[0] = 0
            # elif random_value1 < 2.0 / 3.0:
            #     user_ctr = 1
            #     user_type[0] = 1
            # else:
            #     user_ctr = 1.2
            #     user_type[0] = 2
            #     # 决定用户转化率
            # if random_value2 < 1.0 / 3.0:
            #     user_cvr = 0.8
            #     user_type[1] = 0
            # elif random_value2 < 2.0 / 3.0:
            #     user_cvr = 1
            #     user_type[1] = 1
            # else:
            #     user_cvr = 1.2
            #     user_type[1] = 2
            #     # 决定用户收入水平
            # if random_value3 < 1.0 / 3.0:
            #     user_gmv = 0.8
            #     user_type[2] = 0
            # elif random_value3 < 2.0 / 3.0:
            #     user_gmv = 1
            #     user_type[2] = 1
            # else:
            #     user_gmv = 1.2
            #     user_type[2] = 2
                
            user_type[0] = np.random.randint(0,10)
            user_ctr =  (user_type[0]-5)*0.005+0.027
            user_type[1] = np.random.randint(0,10)
            user_cvr =  (user_type[1]-5)*0.01+0.17
            user_type[2] = np.random.randint(0,10)
            user_gmv =  (user_type[2]-5)*1+40
                # 生成用户对各类型商家的喜好
            for type in range(Number_type_of_store):
                store_prefer[type] = random.uniform(0, 10)
            store_prefer_sum = store_prefer.sum()
            for type in range(Number_type_of_store):
                store_prefer[type] = store_prefer[type] / store_prefer_sum
            # 生成用户对各渠道的喜好并决定当前渠道类型
            for type in range(Number_of_channels):
                channel_prefer[type] = random.uniform(0, 10)
            channel_prefer_sum = channel_prefer.sum()
            current_channel_prefer = 0
            for type in range(Number_of_channels):
                channel_prefer[type] = channel_prefer[type] / channel_prefer_sum
                current_channel_prefer += channel_prefer[type]
                channel_sum[type] = current_channel_prefer

            current_channel = 0
            random_value4 = random.uniform(0, 1)
            for type in range(Number_of_channels):
                if random_value4 < channel_sum[type]:
                    current_channel = type
                    break
                current_channel = Number_of_channels - 1
            today_chan_pv[current_channel] += 1    #被触发渠道当天的pv数增加
            # 随机选取适量的商家,注意保证商家选取不重复
            flag = 0
            for store in range(Number_advertisers_for_pv):
                stores_for_pv[store] = np.random.randint(Number_advertisers)
                for i in range(store):
                    if stores_for_pv[store] == stores_for_pv[i]:
                        flag = 1  # 重复了或选择的商家剩余预算为0就重新选一次
                if store_budget_leave[int(stores_for_pv[store])] <= 0:
                    flag = 1
                if flag == 1:
                    store -= 1
                    flag = 0
            # 每个商家随机出价,注意在当前渠道的预算限制
            for store in range(Number_advertisers_for_pv):
                store_bid[store] = user_ctr*store_CPC[store]               
                #store_bid[store] = random.uniform(Min_bid, Max_bid)    #（0， 1）
                if store_budget_leave[int(stores_for_pv[store])] < store_bid[store]:
                    store_bid[store] = store_budget_leave[int(stores_for_pv[store])]
                #每个商家在次渠道的参与次数+1
                today_bidn_in_chan[int(stores_for_pv[store]), current_channel] += 1


            # 出价结束，选定胜者，胜者方修改预算
            store_bid_list = store_bid.tolist()
            list_max = max(store_bid_list)
            max_index = store_bid_list.index(max(store_bid_list))  # 出价最高的商家在参与竞价的商家中的位置
            temp = store_bid.copy()
            # np.delete(temp, max_index)
            temp = np.delete(temp, max_index)
            store_budget_leave[int(stores_for_pv[max_index])] -= np.max(temp)  # 获胜商家的预算减去第二高的出价作为成本
            attend_bid_time[int(stores_for_pv[max_index])] += 1  # 获胜商家竞价成功次数加一
            today_winn_in_chan[int(stores_for_pv[max_index]), current_channel] += 1    #获胜商家在该渠道的竞价成功数加1
            today_cost_in_chan[int(stores_for_pv[max_index]), current_channel] += np.max(temp)    #获胜商家在该渠道的成本增加第二高出价


            # 开始用户行为判定
            random_ctr_value = np.random.rand()*0.027*10*Number_type_of_store
            random_cvr_value = np.random.rand()*0.17*10*Number_type_of_store
            current_gmv = 0
            all_ctr = store_CTR[int(stores_for_pv[max_index])] * user_ctr * store_prefer[
                int(store_type[int(stores_for_pv[max_index])])] * Number_type_of_store  # 最终用户点击率=商家点击率*用户点击率*用户对该商家类型的喜爱程度*商家类型数
            all_cvr = store_CVR[int(stores_for_pv[max_index])] * user_cvr * store_prefer[
                int(store_type[int(stores_for_pv[max_index])])] * Number_type_of_store  # 最终用户转化率=商家转化率*用户转化率*用户对该商家类型的喜爱程度*商家类型数
            if random_ctr_value <= np.random.normal(all_ctr, 0.05):  # 用户是否点击与是否成交（使用以最终点击率、成交率和成交额为均值的正态分布）
                bool_click = 1    #点击发生
                today_clickn_in_chan[int(stores_for_pv[max_index]), current_channel] += 1    #获胜商家在该渠道的点击数加1
                if random_cvr_value <= np.random.normal(all_cvr, 0.05):
                    bool_convert = 1    #发生转化
                    today_gmvn_in_chan[int(stores_for_pv[max_index]), current_channel] += 1    #获胜商家在该渠道的成交数加1
                    all_gmv = store_GMV[int(stores_for_pv[max_index])] * user_gmv * store_prefer[int(store_type[
                        int(stores_for_pv[max_index])])] * Number_type_of_store  # 最终用户成交额=商家成交额*用户成交额*用户对该商家类型的喜爱程度*商家类型数
                    current_gmv = np.random.normal(all_gmv, 0.05)
                    today_gmv_in_chan[int(stores_for_pv[max_index]), current_channel] += current_gmv    #获胜商家在该渠道的成交额增加
            # 修改商家的平均CTR,CVR和GMV
            for store in range(Number_advertisers_for_pv):
                if store == max_index:  # 针对获胜商家
                    if bool_click == 1:  # 点击了，新的商家点击率={[商家点击率*（100+商家获胜次数）]+1}/（101+商家获胜次数）
                        store_CTR[int(stores_for_pv[store])] = ((store_CTR[int(stores_for_pv[store])] * (
                                100 + attend_bid_time[int(stores_for_pv[store])])) + 1.0) / (
                                                                  101 + attend_bid_time[int(stores_for_pv[store])])
                        if bool_convert == 1:  # 成交了，新的商家成交率={[商家转化率*（100+商家获胜次数）]+1}/（101+商家获胜次数）
                            store_CVR[int(stores_for_pv[store])] = ((store_CVR[int(stores_for_pv[store])] * (
                                    100 + attend_bid_time[int(stores_for_pv[store])])) + 1.0) / (
                                                                      101 + attend_bid_time[int(stores_for_pv[store])])
                            store_GMV[int(stores_for_pv[store])] = ((store_GMV[int(stores_for_pv[store])] * (
                                    100 + attend_bid_time[int(stores_for_pv[store])])) + current_gmv) / (
                                                                      101 + attend_bid_time[int(stores_for_pv[store])])
                            # 成交了，新的商家成交额={[商家成交额*（100+商家获胜次数）]+本次成交额}/（101+商家获胜次数）
                        else:  # 没成交，新的商家成交率={[商家转化率*（100+商家获胜次数）]}/（101+商家获胜次数）
                            store_CVR[int(stores_for_pv[store])] = ((store_CVR[int(stores_for_pv[store])] * (
                                    100 + attend_bid_time[int(stores_for_pv[store])]))) / (
                                                                      101 + attend_bid_time[int(stores_for_pv[store])])
                    else:  # 没点击，新的商家点击率={[商家点击率*（100+商家获胜次数）]}/（101+商家获胜次数）
                        store_CTR[int(stores_for_pv[store])] = ((store_CTR[int(stores_for_pv[store])] * (
                                100 + attend_bid_time[int(stores_for_pv[store])]))) / (
                                                                  101 + attend_bid_time[int(stores_for_pv[store])])
        # 一天结束，所有商家记录当天的历史效益。
        all_hist_gmv_in_chan[day, :] = today_gmv_in_chan
        all_hist_cost_in_chan[day, :] = today_cost_in_chan
        all_hist_gmvn_in_chan[day, :] = today_gmvn_in_chan
        all_hist_clickn_in_chan[day, :] = today_clickn_in_chan
        all_hist_winn_in_chan[day, :] = today_winn_in_chan
        all_hist_bidn_in_chan[day, :] = today_bidn_in_chan
        today_bidn_in_chan_temp = today_bidn_in_chan * 1    #作为除数不能有0
        today_bidn_in_chan_temp[today_bidn_in_chan_temp==0] = 1
        today_winn_in_chan_temp = today_winn_in_chan * 1
        today_winn_in_chan_temp[today_winn_in_chan_temp==0] = 1
        today_clickn_in_chan_temp = today_clickn_in_chan * 1
        today_clickn_in_chan_temp[today_clickn_in_chan_temp == 0] = 1
        today_cost_in_chan_temp = today_cost_in_chan * 1
        today_cost_in_chan_temp[today_cost_in_chan_temp == 0] = 1
        all_hist_winr_in_chan[day, :] = today_winn_in_chan / today_bidn_in_chan_temp
        all_hist_ctr_in_chan[day, :] = today_clickn_in_chan / today_winn_in_chan_temp
        all_hist_cvr_in_chan[day, :] = today_gmvn_in_chan / today_clickn_in_chan_temp
        all_hist_bud_in_chan[day, :] = store_budget_in_chan
        all_hist_roi_in_chan[day, :] = store_ROI_in_chan
        all_hist_real_roi_in_chan[day: ] = today_gmv_in_chan / today_cost_in_chan_temp
        all_hist_chan_pv[day] = today_chan_pv
        all_hist_roi[day] = store_ROI

    # 正式统计过程开始
    # 高层状态信息：历史H天内，每天第i个渠道获得了几次pv，商家在第ｉ个渠道每天的成交额、成本、成交数、点击数、竞价成功数、竞价总参与数、每天的竞价成功率、点击率、转化率、预算、目标ROI参考值、真实ROI，每天所有渠道的总pv数，商家每天在所有渠道的总成交额、总成本、总成交数、总点击数、总竞价成功数、总竞价参与数、综合竞价成功率、综合点击率、综合转化率、总预算、总目标ROI、总真实ROI，商家当天的预算和ROI
    # 高层动作信息：分配预算参考值权重，分配GMV参考值权重（计算，最后可得预算ROI参考值）
    # 高层奖励信息：商家在所有渠道的综合点击率
    # state_h[0, :] = np.concatenate((
    #                   [np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),
    #                   np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),
    #                   np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),
    #                   np.zeros(Number_days),]*Number_of_channels,
    #                   np.zeros(Number_days),
    #                   np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),
    #                   np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),
    #                   np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),np.zeros(Number_days),
    #                   np.zero(1),np.zero(1),))
    # state_h = np.zeros([Number_days*Number_advertisers,Number_days*13*Number_of_channels+Number_days*13+2])
    # 低层状态信息：
    # 商家在本渠道信息：商家编号，当天第几个pv, 当前是竞价的第几天，当前渠道，这是当前渠道的第几个pv，
    #               用户类型，用户对商家类型的喜好分布，用户对渠道的喜好分布，商家类型，商家平均CTR/CVR/GMV，
    #               商家当天当前在该渠道的成交额、成本、成交数、点击数、竞价成功数、竞价总参与数、
    #               当天的竞价成功率、点击率、转化率、预算参考值、GＭV参考值、ROI参考值、相对于预算参考值的剩余金额、已达成ROI
    # 其他渠道的信息：第i个渠道获得了几次pv，商家在第i个渠道的成交额、成本、成交数、点击数、竞价成功数、竞价总参与数、
    #               当天的竞价成功率、点击率、转化率、预算参考值、GＭV参考值、ROI参考值、相对于预算参考值的剩余金额、已达成ROI
    # 商家综合信息：商家在所有渠道的总成交额、总成本、总成交数、总点击数、总竞价成功数、总竞价参与数、
    #           综合竞价成功率、综合点击率、综合转化率、总预算、总ROI，商家剩余预算，商家当前已达成的ROI
    # state_l[0, :] = np.concatenate((
    #                   np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),
    #                   np.zeros(3),np.zeros(Number_type_of_store),np.zeros(Number_of_channels),
    #                   np.zeros(1),np.zeros(1),np.zeros(1),np.zeros(1),     类型、CTR、CVR、GMV
    #                   np.zeros(14),
    #                   np.zeros(15)*Number_of_channels,
    #                   np.zeros(13),)
    # state_l = np.zeros([Number_days*Number_pvs*Number_advertisers_for_pv,
    #                   12+14+15*Number_of_channels+13+Number_type_of_store+Number_of_channels])

    state_h_dim = Number_days*13*Number_of_channels+Number_days*13+2    # 158
    state_l_dim = 13+14+15*(Number_of_channels-1)+13+Number_type_of_store+Number_of_channels    #
    # 初始化高层状态，动作，奖励、成本，下一个状态，是否终止
    state_h = np.zeros([Number_advertisers, Number_days, state_h_dim])
    action_b_h = np.zeros([Number_advertisers, Number_days, Number_of_channels])
    action_g_h = np.zeros([Number_advertisers, Number_days, Number_of_channels])
    reward_h = np.zeros([Number_advertisers, Number_days])
    next_state_h = np.zeros([Number_advertisers, Number_days, state_h_dim])
    terminal_h = np.zeros([Number_advertisers, Number_days])
    for day in range(Number_days):
        # 初始化低层状态，动作，奖励、成本、GMV、商家ROI（作为约束的值），下一个状态，是否终止
        # Number_advertisers行，Number_of_channels列个空列表[]
        state_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        action_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        reward_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        cost_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        gmv_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        roi_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        next_state_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        terminal_l = [[[] for j in range(Number_of_channels)] for i in range(Number_advertisers)]
        # 初始化，当天每个商家在每个渠道上的成交额，成本，成交数，点击数，竞价成功数，参与竞价次数
        today_gmv_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_cost_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_gmvn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_clickn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_winn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_bidn_in_chan = np.zeros([Number_advertisers, Number_of_channels])
        today_chan_pv = np.zeros(Number_of_channels)  # 当天各渠道总pv
        # 每个商家为每个渠道分配当天的预算和ROI
        store_budget = np.zeros(Number_advertisers)  # 初始化，每个商家当天的预算
        store_budget_leave = np.zeros(Number_advertisers)  # 初始化，商家竞价时的剩余预算，最初等于商家当天的预算
        store_ROI = np.zeros(Number_advertisers)  # 初始化，每个商家当天的ROI
        store_budget_in_chan = np.zeros([Number_advertisers, Number_of_channels])  # 初始化，每个商家在每个渠道当天的预算参考值
        store_gmv_in_chan = np.zeros([Number_advertisers, Number_of_channels])  # 初始化，每个商家在每个渠道当天的GMV参考值
        store_ROI_in_chan = np.zeros([Number_advertisers, Number_of_channels])  # 初始化，每个商家在每个渠道当天的ROI参考值
        for store in range(Number_advertisers):
            store_budget[store] = random.uniform(Min_budget, Max_budget)  # 预算值多少比较好
            store_ROI[store] = random.uniform(Min_ROI, Max_ROI)  # ROI值多少比较好
            budget_weight = np.random.random(Number_of_channels)  # 给各渠道的预算参考值权重
            flag = True  # 非零处理
            while flag:
                flag = False
                for c in range(Number_of_channels):
                    if budget_weight[c] == 0.0:
                        flag = True
                        budget_weight[c] = random.uniform(0, 1)
            budget_a = budget_weight / budget_weight.sum()
            store_budget_in_chan[store] = budget_a * store_budget[store]
            gmv_weight = np.random.random(Number_of_channels)  # 给各渠道的GMV参考值权重
            gmv_a = gmv_weight / gmv_weight.sum()
            store_gmv_in_chan[store] = gmv_a * store_budget[store] * store_ROI[store]
            # store_ROI_in_chan[store] = [store_gmv_in_chan[store][chan] / store_budget_in_chan[store][chan] for chan in range(Number_of_channels)]  # 给各渠道的ROI参考值
            store_ROI_in_chan[store] = store_gmv_in_chan[store] / store_budget_in_chan[store]  # 给各渠道的ROI参考值
            # 记录状态动作
            chan_state_h = np.array([])
            all_hist_cost_in_chan_temp = all_hist_cost_in_chan * 1
            all_hist_cost_in_chan_temp[all_hist_cost_in_chan_temp == 0] = 1
            for chan in range(Number_of_channels):
                chan_state_h = np.concatenate((chan_state_h,
                                               all_hist_chan_pv[:, chan], all_hist_gmv_in_chan[:, store, chan],
                                               all_hist_cost_in_chan[:, store, chan], all_hist_gmvn_in_chan[:, store, chan],
                                               all_hist_clickn_in_chan[:, store, chan], all_hist_winn_in_chan[:, store, chan],
                                               all_hist_bidn_in_chan[:, store, chan], all_hist_winr_in_chan[:, store, chan],
                                               all_hist_ctr_in_chan[:, store, chan], all_hist_cvr_in_chan[:, store, chan],
                                               all_hist_bud_in_chan[:, store, chan], all_hist_roi_in_chan[:, store, chan],
                                               all_hist_gmv_in_chan[:, store, chan]/all_hist_cost_in_chan_temp[:, store, chan]))
            all_hist_bidn_temp = all_hist_bidn_in_chan[:, store, :].sum(axis=1) * 1  # 作为除数不能有0
            all_hist_bidn_temp[all_hist_bidn_temp == 0] = 1
            all_hist_winn_temp = all_hist_winn_in_chan[:, store, :].sum(axis=1) * 1
            all_hist_winn_temp[all_hist_winn_temp == 0] = 1
            all_hist_clickn_temp = all_hist_clickn_in_chan[:, store, :].sum(axis=1) * 1
            all_hist_clickn_temp[all_hist_clickn_temp == 0] = 1
            all_hist_cost_temp = all_hist_cost_in_chan[:, store, :].sum(axis=1) * 1
            all_hist_cost_temp[all_hist_cost_temp == 0] = 1
            state_h[store, day, :] = np.concatenate((chan_state_h, all_hist_chan_pv.sum(axis=1), all_hist_gmv_in_chan[:, store, :].sum(axis=1),
                                                     all_hist_cost_in_chan[:, store, :].sum(axis=1), all_hist_gmvn_in_chan[:, store, :].sum(axis=1),
                                                     all_hist_clickn_in_chan[:, store, :].sum(axis=1), all_hist_winn_in_chan[:, store, :].sum(axis=1),
                                                     all_hist_bidn_in_chan[:, store, :].sum(axis=1), all_hist_winn_in_chan[:, store, :].sum(axis=1)/all_hist_bidn_temp,
                                                     all_hist_clickn_in_chan[:, store, :].sum(axis=1)/all_hist_winn_temp, all_hist_gmvn_in_chan[:, store, :].sum(axis=1)/all_hist_clickn_temp,
                                                     all_hist_bud_in_chan[:, store, :].sum(axis=1), all_hist_roi[:, store],
                                                     all_hist_gmv_in_chan[:, store, :].sum(axis=1)/all_hist_cost_temp,
                                                     np.ones(1)*store_budget[store], np.ones(1)*store_ROI[store]))
            # 动作，两部分
            action_b_h[store, day, :] = budget_a
            action_g_h[store, day, :] = gmv_a
            # # 下一个状态以及是否结束
            # if (day + 1) % High_cycle_days != 1:
            #     next_state_h[store, day - 1, :] = state_h[store, day, :]
            # if (day + 1) % High_cycle_days == 0:
            #     terminal_h[store, day] = 1
            # 下一个状态以及是否结束
            if day > 0:
                next_state_h[store, day - 1, :] = state_h[store, day, :]
            if (day + 1) == Number_days:
                terminal_h[store, day] = 1
        store_budget_leave = store_budget * 1

        for pv in range(Number_pvs):    # 遍历当天每一条pv流量
            # print('formal_pv: ', pv)
            print('formal_day: ', day, '......', 'formal_pv: ', pv)
            user_type = np.zeros(3)    # 初始化，当前用户在点击率，转化率，成交额方面的类型
            bool_click = 0  # 是否点击
            bool_convert = 0  # 是否转化
            store_prefer = np.zeros(Number_type_of_store)  # 初始化，维度为商家类型数量的先验，代表当前用户对各类型商家的喜好程度（喜欢烧烤、面食等情况）
            channel_prefer = np.zeros(Number_of_channels)  # 初始化，维度为渠道数量的先验，代表当前用户对各渠道的喜爱程度
            channel_sum = np.zeros(Number_of_channels)  # 初始化，维度为渠道数量，代表[渠道一喜爱程度,渠道一加二喜爱程度,渠道一加二加三喜爱程度,...]
            stores_for_pv = np.zeros(Number_advertisers_for_pv)  # 初始化，维度为pv商家数，代表参与当前pv流量竞价的商家
            store_bid = np.zeros(Number_advertisers_for_pv)  # 初始化，维度为pv商家数，代表参与竞价的商家出的价
            type_stores_for_pv = np.zeros(Number_advertisers_for_pv)
            ctr_for_pv = np.zeros(Number_advertisers_for_pv)
            cvr_for_pv = np.zeros(Number_advertisers_for_pv)
            gmv_for_pv = np.zeros(Number_advertisers_for_pv)
            today_gmv_for_pv = np.zeros(Number_advertisers_for_pv)
            today_cost_for_pv = np.zeros(Number_advertisers_for_pv)
            today_gmvn_for_pv = np.zeros(Number_advertisers_for_pv)
            today_clickn_for_pv = np.zeros(Number_advertisers_for_pv)
            today_winn_for_pv = np.zeros(Number_advertisers_for_pv)
            today_bidn_for_pv = np.zeros(Number_advertisers_for_pv)
            today_winr_for_pv = np.zeros(Number_advertisers_for_pv)
            today_ctr_for_pv = np.zeros(Number_advertisers_for_pv)
            today_cvr_for_pv = np.zeros(Number_advertisers_for_pv)
            today_bud_for_pv = np.zeros(Number_advertisers_for_pv)
            today_roi_for_pv = np.zeros(Number_advertisers_for_pv)
            today_bud_cur_for_pv = np.zeros(Number_advertisers_for_pv)
            today_roi_cur_for_pv = np.zeros(Number_advertisers_for_pv)
            # random_value1 = np.random.rand()
            # random_value2 = np.random.rand()
            # random_value3 = np.random.rand()
            # 决定用户点击率
            # if random_value1 < 1.0 / 3.0:
            #     user_ctr = 0.027*0.8
            #     user_type[0] = 0
            # elif random_value1 < 2.0 / 3.0:
            #     user_ctr = 1
            #     user_type[0] = 1
            # else:
            #     user_ctr = 1.2
            #     user_type[0] = 2
            #     # 决定用户转化率
            # if random_value2 < 1.0 / 3.0:
            #     user_cvr = 0.8
            #     user_type[1] = 0
            # elif random_value2 < 2.0 / 3.0:
            #     user_cvr = 1
            #     user_type[1] = 1
            # else:
            #     user_cvr = 1.2
            #     user_type[1] = 2
            #     # 决定用户收入水平
            # if random_value3 < 1.0 / 3.0:
            #     user_gmv = 0.8
            #     user_type[2] = 0
            # elif random_value3 < 2.0 / 3.0:
            #     user_gmv = 1
            #     user_type[2] = 1
            # else:
            #     user_gmv = 1.2
            #     user_type[2] = 2
                
            user_type[0] = np.random.randint(0,10)
            user_ctr =  (user_type[0]-5)*0.005+0.027
            user_type[1] = np.random.randint(0,10)
            user_cvr =  (user_type[1]-5)*0.05+0.17
            user_type[2] = np.random.randint(0,10)
            user_gmv =  (user_type[2]-5)*1+40
                # 生成用户对各类型商家的喜好
            for type in range(Number_type_of_store):
                store_prefer[type] = random.uniform(0, 10)
            store_prefer_sum = store_prefer.sum()
            for type in range(Number_type_of_store):
                store_prefer[type] = store_prefer[type] / store_prefer_sum
            # 生成用户对各渠道的喜好并决定当前渠道类型
            for type in range(Number_of_channels):
                channel_prefer[type] = random.uniform(0, 10)
            channel_prefer_sum = channel_prefer.sum()
            current_channel_prefer = 0
            for type in range(Number_of_channels):
                channel_prefer[type] = channel_prefer[type] / channel_prefer_sum
                current_channel_prefer += channel_prefer[type]
                channel_sum[type] = current_channel_prefer

            current_channel = 0
            random_value4 = random.uniform(0, 1)
            for type in range(Number_of_channels):
                if random_value4 < channel_sum[type]:
                    current_channel = type
                    break
                current_channel = Number_of_channels - 1
            today_chan_pv[current_channel] += 1  # 被触发渠道当天的pv数增加
            # 随机选取适量的商家,注意保证商家选取不重复
            flag = 0
            for store in range(Number_advertisers_for_pv):
                stores_for_pv[store] = np.random.randint(Number_advertisers)
                for i in range(store):
                    if stores_for_pv[store] == stores_for_pv[i]:
                        flag = 1  # 重复了或选择的商家剩余预算为0就重新选一次
                if store_budget_leave[int(stores_for_pv[store])] <= 0:
                    flag = 1
                if flag == 1:
                    store -= 1
                    flag = 0

            # 记录状态
            for store in range(Number_advertisers_for_pv):
                store_pv = stores_for_pv[store]
                # 商家在本渠道信息
                cur_for_gmv = today_gmv_in_chan[int(store_pv), current_channel]
                cur_for_cost = today_cost_in_chan[int(store_pv), current_channel]
                cur_for_gmvn = today_gmvn_in_chan[int(store_pv), current_channel]
                cur_for_clickn = today_clickn_in_chan[int(store_pv), current_channel]
                cur_for_winn = today_winn_in_chan[int(store_pv), current_channel]
                cur_for_bidn = today_bidn_in_chan[int(store_pv), current_channel]
                cur_for_winr = 0 if cur_for_bidn == 0 else cur_for_winn / cur_for_bidn
                cur_for_ctr = 0 if cur_for_winn == 0 else cur_for_clickn / cur_for_winn
                cur_for_cvr = 0 if cur_for_clickn == 0 else cur_for_gmvn / cur_for_clickn
                cur_for_budget_ref = store_budget_in_chan[int(store_pv), current_channel]
                cur_for_gmv_ref = store_gmv_in_chan[int(store_pv), current_channel]
                cur_for_roi_ref = store_ROI_in_chan[int(store_pv), current_channel]
                cur_for_roi_real = 0 if cur_for_cost == 0 else cur_for_gmv / cur_for_cost
                current_channel_state_l = np.concatenate((
                    np.array([store_pv, pv, day + 1, current_channel, today_chan_pv[current_channel]]),
                    user_type, store_prefer, channel_prefer, np.ones(1)*store_type[int(store_pv)],
                    np.array([store_CPC[int(store_pv)], store_CTR[int(store_pv)], store_CVR[int(store_pv)], store_GMV[int(store_pv)]]),
                    np.array([cur_for_gmv, cur_for_cost, cur_for_gmvn, cur_for_clickn, cur_for_winn,
                              cur_for_bidn, cur_for_winr, cur_for_ctr, cur_for_cvr, cur_for_budget_ref,
                              cur_for_gmv_ref, cur_for_roi_ref, cur_for_budget_ref-cur_for_cost,
                              cur_for_roi_real])
                ))
                # 其他渠道的信息
                other_channel_state_l = np.array([])
                for chan_state in range(Number_of_channels):
                    if chan_state != current_channel:
                        oth_for_gmv = today_gmv_in_chan[int(store_pv), chan_state]
                        oth_for_cost = today_cost_in_chan[int(store_pv), chan_state]
                        oth_for_gmvn = today_gmvn_in_chan[int(store_pv), chan_state]
                        oth_for_clickn = today_clickn_in_chan[int(store_pv), chan_state]
                        oth_for_winn = today_winn_in_chan[int(store_pv), chan_state]
                        oth_for_bidn = today_bidn_in_chan[int(store_pv), chan_state]
                        oth_for_winr = 0 if oth_for_bidn == 0 else oth_for_winn / oth_for_bidn
                        oth_for_ctr = 0 if oth_for_winn == 0 else oth_for_clickn / oth_for_winn
                        oth_for_cvr = 0 if oth_for_clickn == 0 else oth_for_gmvn / oth_for_clickn
                        oth_for_budget_ref = store_budget_in_chan[int(store_pv), chan_state]
                        oth_for_gmv_ref = store_gmv_in_chan[int(store_pv), chan_state]
                        oth_for_roi_ref = store_ROI_in_chan[int(store_pv), chan_state]
                        oth_for_roi_real = 0 if oth_for_cost == 0 else oth_for_gmv / oth_for_cost
                        other_channel_state_l = np.concatenate((
                            other_channel_state_l, np.ones(1)*today_chan_pv[chan_state],
                            np.array([oth_for_gmv, oth_for_cost, oth_for_gmvn, oth_for_clickn, oth_for_winn,
                                      oth_for_bidn, oth_for_winr, oth_for_ctr, oth_for_cvr, oth_for_budget_ref,
                                      oth_for_gmv_ref, oth_for_roi_ref, oth_for_budget_ref - oth_for_cost,
                                      oth_for_roi_real])
                        ))
                # 商家综合信息
                total_for_gmv = today_gmv_in_chan[int(store_pv), :].sum()
                total_for_cost = today_cost_in_chan[int(store_pv), :].sum()
                total_for_gmvn = today_gmvn_in_chan[int(store_pv), :].sum()
                total_for_clickn = today_clickn_in_chan[int(store_pv), :].sum()
                total_for_winn = today_winn_in_chan[int(store_pv), :].sum()
                total_for_bidn = today_bidn_in_chan[int(store_pv), :].sum()
                total_for_winr = 0 if total_for_bidn == 0 else total_for_winn / total_for_bidn
                total_for_ctr = 0 if total_for_winn == 0 else total_for_clickn / total_for_winn
                total_for_cvr = 0 if total_for_clickn == 0 else total_for_gmvn / total_for_clickn
                total_for_budget = store_budget[int(store_pv)]
                total_for_roi = store_ROI[int(store_pv)]
                total_for_budget_leave = store_budget_leave[int(store_pv)]
                total_for_roi_real = 0 if total_for_cost == 0 else total_for_gmv / total_for_cost
                total_channel_state_l = np.array([
                    total_for_gmv, total_for_cost, total_for_gmvn, total_for_clickn, total_for_winn,
                    total_for_bidn, total_for_winr, total_for_ctr, total_for_cvr, total_for_budget,
                    total_for_roi, total_for_budget_leave, total_for_roi_real
                ])
                # 将三者连起
                state_l_one_store = np.concatenate((current_channel_state_l, other_channel_state_l, total_channel_state_l))
                state_l[int(store_pv)][current_channel].append(state_l_one_store.tolist())

                # 出价，并记录出价动作（是否需对比当前剩余预算？？？）
                action_bid =  user_ctr*store_CPC[store]  
                #action_bid = random.uniform(Min_bid, Max_bid)
                if store_budget_leave[int(store_pv)] < action_bid:
                    action_bid = store_budget_leave[int(store_pv)]
                action_l[int(store_pv)][current_channel].append(action_bid)
                store_bid[store] = action_bid

                # 默认奖励暂时为竞价失败的-1
                reward_l[int(store_pv)][current_channel].append(0)
                # 默认成本暂时为竞价失败的0
                cost_l[int(store_pv)][current_channel].append(0)
                # 默认gmv暂时为0
                gmv_l[int(store_pv)][current_channel].append(0)
                # 记录ROI为该商家当天的总目标ROI
                roi_l[int(store_pv)][current_channel].append(store_ROI[int(store_pv)])
                # 记录下一个动作
                store_state_count = len(action_l[int(store_pv)][current_channel])    #此时该商家的状态动作数量
                if store_state_count > 1:
                    next_state_l[int(store_pv)][current_channel].append(state_l[int(store_pv)][current_channel][-1])
                # 记录终止状态暂时均为0
                terminal_l[int(store_pv)][current_channel].append(0)
                # 每个商家在此渠道的参与次数+1
                today_bidn_in_chan[int(store_pv), current_channel] += 1


            # 出价结束，选定胜者，胜者方修改预算
            store_bid_list = store_bid.tolist()
            list_max = max(store_bid_list)
            max_index = store_bid_list.index(max(store_bid_list))  # 出价最高的商家在参与竞价的商家中的位置
            temp = store_bid.copy()
            # np.delete(temp, max_index)
            temp = np.delete(temp, max_index)
            store_budget_leave[int(stores_for_pv[max_index])] -= np.max(temp)  # 获胜商家的预算减去第二高的出价作为成本
            attend_bid_time[int(stores_for_pv[max_index])] += 1  # 获胜商家竞价成功次数加一
            today_winn_in_chan[int(stores_for_pv[max_index]), current_channel] += 1  # 获胜商家在该渠道的竞价成功数加1
            today_cost_in_chan[int(stores_for_pv[max_index]), current_channel] += np.max(temp)  # 获胜商家在该渠道的成本增加第二高出价
            # 获胜商家的成本修改
            cost_l[int(stores_for_pv[max_index])][current_channel][-1] = np.max(temp)


            # 开始用户行为判定
            k = 0
            current_gmv = 0
            random_ctr_value = np.random.rand()*0.027*10*Number_type_of_store
            random_cvr_value = np.random.rand()*0.17*10*Number_type_of_store
            all_ctr = store_CTR[int(stores_for_pv[max_index])] * user_ctr * store_prefer[
                int(store_type[int(stores_for_pv[max_index])])] * Number_type_of_store  # 最终用户点击率=商家点击率*用户点击率*用户对该商家类型的喜爱程度*商家类型数
            all_cvr = store_CVR[int(stores_for_pv[max_index])] * user_cvr * store_prefer[
                int(store_type[int(stores_for_pv[max_index])])] * Number_type_of_store  # 最终用户转化率=商家转化率*用户转化率*用户对该商家类型的喜爱程度*商家类型数
            if random_ctr_value <= np.random.normal(all_ctr, 0.05):  # 用户是否点击与是否成交（使用以最终点击率、成交率和成交额为均值的正态分布）
                bool_click = 1    #点击发生
                today_clickn_in_chan[int(stores_for_pv[max_index]), current_channel] += 1    #获胜商家在该渠道的点击数加1
                # k = 0.5
                k = 1    #点击则奖励为1
                if random_cvr_value <= np.random.normal(all_cvr, 0.05):
                    bool_convert = 1    #发生转化
                    # k = 0.8
                    today_gmvn_in_chan[int(stores_for_pv[max_index]), current_channel] += 1    #获胜商家在该渠道的成交数加1
                    all_gmv = store_GMV[int(stores_for_pv[max_index])] * user_gmv * store_prefer[int(store_type[
                        int(stores_for_pv[max_index])])] * Number_type_of_store  # 最终用户成交额=商家成交额*用户成交额*用户对该商家类型的喜爱程度*商家类型数
                    current_gmv = np.random.normal(all_gmv, 0.05)
                    today_gmv_in_chan[int(stores_for_pv[max_index]), current_channel] += current_gmv    #获胜商家在该渠道的成交额增加
                    # k += delta * current_gmv
            else:
                # k = 0.2
                k = 0
            # 记录获证商家的奖励和gmv
            reward_l[int(stores_for_pv[max_index])][current_channel][-1] = k
            gmv_l[int(stores_for_pv[max_index])][current_channel][-1] = current_gmv

            # 修改商家的平均CTR,CVR和GMV
            for store in range(Number_advertisers_for_pv):
                if store == max_index:  # 针对获胜商家
                    if bool_click == 1:  # 点击了，新的商家点击率={[商家点击率*（100+商家获胜次数）]+1}/（101+商家获胜次数）
                        store_CTR[int(stores_for_pv[store])] = ((store_CTR[int(stores_for_pv[store])] * (
                                100 + attend_bid_time[int(stores_for_pv[store])])) + 1.0) / (
                                                                  101 + attend_bid_time[int(stores_for_pv[store])])
                        if bool_convert == 1:  # 成交了，新的商家成交率={[商家转化率*（100+商家获胜次数）]+1}/（101+商家获胜次数）
                            store_CVR[int(stores_for_pv[store])] = ((store_CVR[int(stores_for_pv[store])] * (
                                    100 + attend_bid_time[int(stores_for_pv[store])])) + 1.0) / (
                                                                      101 + attend_bid_time[int(stores_for_pv[store])])
                            store_GMV[int(stores_for_pv[store])] = ((store_GMV[int(stores_for_pv[store])] * (
                                    100 + attend_bid_time[int(stores_for_pv[store])])) + current_gmv) / (
                                                                      101 + attend_bid_time[int(stores_for_pv[store])])
                            # 成交了，新的商家成交额={[商家成交额*（100+商家获胜次数）]+本次成交额}/（101+商家获胜次数）
                        else:  # 没成交，新的商家成交率={[商家转化率*（100+商家获胜次数）]}/（101+商家获胜次数）
                            store_CVR[int(stores_for_pv[store])] = ((store_CVR[int(stores_for_pv[store])] * (
                                    100 + attend_bid_time[int(stores_for_pv[store])]))) / (
                                                                      101 + attend_bid_time[int(stores_for_pv[store])])
                    else:  # 没点击，新的商家点击率={[商家点击率*（100+商家获胜次数）]}/（101+商家获胜次数）
                        store_CTR[int(stores_for_pv[store])] = ((store_CTR[int(stores_for_pv[store])] * (
                                100 + attend_bid_time[int(stores_for_pv[store])]))) / (
                                                                  101 + attend_bid_time[int(stores_for_pv[store])])
        # 一天结束，所有商家记更新当天的历史效益。

        all_hist_gmv_in_chan = np.concatenate((all_hist_gmv_in_chan, today_gmv_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_gmv_in_chan = np.delete(all_hist_gmv_in_chan, 0, 0)
        all_hist_cost_in_chan = np.concatenate((all_hist_cost_in_chan, today_cost_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_cost_in_chan = np.delete(all_hist_cost_in_chan, 0, 0)
        all_hist_gmvn_in_chan = np.concatenate((all_hist_gmvn_in_chan, today_gmvn_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_gmvn_in_chan = np.delete(all_hist_gmvn_in_chan, 0, 0)
        all_hist_clickn_in_chan = np.concatenate((all_hist_clickn_in_chan, today_clickn_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_clickn_in_chan = np.delete(all_hist_clickn_in_chan, 0, 0)
        all_hist_winn_in_chan = np.concatenate((all_hist_winn_in_chan, today_winn_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_winn_in_chan = np.delete(all_hist_winn_in_chan, 0, 0)
        all_hist_bidn_in_chan = np.concatenate((all_hist_bidn_in_chan, today_bidn_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_bidn_in_chan = np.delete(all_hist_bidn_in_chan, 0, 0)
        today_bidn_in_chan_temp = today_bidn_in_chan * 1    #作为除数不能有0
        today_bidn_in_chan_temp[today_bidn_in_chan_temp==0] = 1
        today_winn_in_chan_temp = today_winn_in_chan * 1
        today_winn_in_chan_temp[today_winn_in_chan_temp==0] = 1
        today_clickn_in_chan_temp = today_clickn_in_chan * 1
        today_clickn_in_chan_temp[today_clickn_in_chan_temp == 0] = 1
        all_hist_winr_in_chan = np.concatenate((all_hist_winr_in_chan, (today_winn_in_chan / today_bidn_in_chan_temp).reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_winr_in_chan = np.delete(all_hist_winr_in_chan, 0, 0)
        all_hist_ctr_in_chan = np.concatenate((all_hist_ctr_in_chan,(today_clickn_in_chan / today_winn_in_chan_temp).reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_ctr_in_chan = np.delete(all_hist_ctr_in_chan, 0, 0)
        all_hist_cvr_in_chan = np.concatenate((all_hist_cvr_in_chan,(today_gmvn_in_chan / today_clickn_in_chan_temp).reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_cvr_in_chan = np.delete(all_hist_cvr_in_chan, 0, 0)
        all_hist_bud_in_chan = np.concatenate((all_hist_bud_in_chan, store_budget_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_bud_in_chan = np.delete(all_hist_bud_in_chan, 0, 0)
        all_hist_roi_in_chan = np.concatenate((all_hist_roi_in_chan, store_ROI_in_chan.reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_roi_in_chan = np.delete(all_hist_roi_in_chan, 0, 0)
        today_cost_in_chan_temp = today_cost_in_chan * 1
        today_cost_in_chan_temp[today_cost_in_chan_temp == 0] = 1
        all_hist_real_roi_in_chan = np.concatenate(
            (all_hist_real_roi_in_chan, (today_gmv_in_chan / today_cost_in_chan_temp).reshape((1, Number_advertisers, Number_of_channels))))
        all_hist_real_roi_in_chan = np.delete(all_hist_real_roi_in_chan, 0, 0)
        all_hist_chan_pv = np.concatenate(
            (all_hist_chan_pv, today_chan_pv.reshape((1, Number_of_channels))))
        all_hist_chan_pv = np.delete(all_hist_chan_pv, 0, 0)
        all_hist_roi = np.concatenate(
            (all_hist_roi, store_ROI.reshape((1, Number_advertisers))))
        all_hist_roi = np.delete(all_hist_roi, 0, 0)
        # print(today_clickn_in_chan)
        # 计算高层的奖励
        for store in range(Number_advertisers):
            today_winn_temp = today_winn_in_chan[store, :].sum()
            today_clickn_temp = today_clickn_in_chan[store, :].sum()
            reward_h[store, day] = 0 if today_winn_temp==0 else today_clickn_temp
            if (day+1) == Number_days:    # 最后一天，终止状态为1
                terminal_h[store, day] = 1

        state_l_day, action_l_day, terminal_l_day, next_state_l_day = [], [], [], []
        # 整理底层数据（遍历每个渠道，设置终止）
        for chan in range(Number_of_channels):
            for store in range(Number_advertisers):
                # 设置终止状态并将终止状态的next_state补充为0向量
                if state_l[store][chan] != []:
                    next_state_l[store][chan].append(np.zeros(state_l_dim).tolist())
                    terminal_l[store][chan][-1] = 1
                # 写入数据
                for l in range(len(state_l[store][chan])):
                    saved_state_l[chan].append(' '.join(str(i) for i in state_l[store][chan][l]))
                    saved_action_l[chan].append(action_l[store][chan][l])
                    saved_reward_l[chan].append(reward_l[store][chan][l])
                    saved_cost_l[chan].append(cost_l[store][chan][l])
                    saved_gmv_l[chan].append(gmv_l[store][chan][l])
                    saved_ROI_l[chan].append(roi_l[store][chan][l])
                    saved_next_state_l[chan].append(' '.join(str(i) for i in next_state_l[store][chan][l]))
                    saved_terminal_l[chan].append(terminal_l[store][chan][l])

                    state_l_day.append(state_l[store][chan][l])
                    action_l_day.append(action_l[store][chan][l])
                    terminal_l_day.append(terminal_l[store][chan][l])
                    next_state_l_day.append(next_state_l[store][chan][l])

        # 把当天的数据state，action，terminal列表存起来，供maddpg的数据查找使用（数组不定长，存不了）（换种方式寻）
        np.savez('trajectory_low_data_Search_day' + str(day+1) + '.npz', state_l_day=state_l_day, action_l_day=action_l_day,
                 terminal_l_day=terminal_l_day)

        print(np.array(state_l_day).shape)
        print(np.array(next_state_l_day).shape)
        print(state_l_dim)

    # 整理高层数据
    for s in range(Number_advertisers):
        for d in range(Number_days):
            saved_state_h.append(' '.join(str(i) for i in state_h[s, d, :]))
            saved_action_b_h.append(' '.join(str(i) for i in action_b_h[s, d, :]))
            saved_action_g_h.append(' '.join(str(i) for i in action_g_h[s, d, :]))
            saved_reward_h.append(reward_h[s, d])
            saved_next_state_h.append(' '.join(str(i) for i in next_state_h[s, d, :]))
            saved_terminal_h.append(terminal_h[s, d])
    #把数据写入文件
    np.savez('trajectory_high.npz',state_h=saved_state_h, action_b_h=saved_action_b_h,
             action_g_h=saved_action_g_h, reward_h=saved_reward_h, next_state_h=saved_next_state_h,
             terminal_h=saved_terminal_h)
    for d in range(Number_of_channels):
        np.savez('trajectory_low_'+str(d)+'.npz', state_l=saved_state_l[d], action_l=saved_action_l[d], reward_l=saved_reward_l[d],
             next_state_l=saved_next_state_l[d], cost_l=saved_cost_l[d], gmv_l=saved_gmv_l[d], roi_l=saved_ROI_l[d],
             terminal_l=saved_terminal_l[d])


if __name__=='__main__':
    args = parser.parse_args()
    main(args)
