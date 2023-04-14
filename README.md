# HiBid


This is the code accompanying the paper: "HiBid: Hierarchical Offline Deep Reinforcement Learning for Cross-Channel Constrained Bidding with Budget Allocation"。
## :page_facing_up: Description
Online display advertising platforms service millions of advertisers by providing real-time bidding (RTB) for billions of ad requests every day. The bidding strategy handles ad requests across channels to maximize the number of clicks under the set financial constraints, i.e., total budget and cost-per-click (CPC), etc. Different from existing works mainly focusing on single channel bidding, we explicitly consider cross-channel constrained bidding with budget allocation. Specifically, we propose a hierarchical offline deep reinforcement learning (DRL) framework called ``HiBid'', based on the state-of-the-art offline DRL approach MCQ with three contributions: (a) auxiliary batch loss for non-competitive budget allocation, (b) 
$\lambda$-generalization for adaptive bidding strategy in response to changing budgets, and (c) CPC-guided action selection to satisfy cross-channel CPC constraint. Through extensive experiments on both the large-scale log data and online A/B testing, we confirm that HiBid outperforms five baselines in terms of the number of clicks, CPC satisfactory ratio, and return-on-investment (ROI). We also deploy HiBid on Meituan advertising platform to already service tens of thousands of advertisers every day. 

### Installation
1. Clone repo
    ```bash
    git clone -c git@github.com:HaoooWang/HiBid.git
    cd HiBid
    ```
   
2. Create Virtual Environment
    ```
   conda create -n hibid python==3.8
   conda activate hibid
   ```
3. Install dependent packages
    ```
    pip install -r requirements.txt
    python setup.py develop
    ```
## :computer: Training

To train HiBid, use:
```
cd /algorithm/HiBid
python main.py --dataset your_dataset_path --use_cpcas --use_bcorle --bcorle_lambda_num 50 --task_type high
```

### 
### Auction Simulator
The synthetic dataset are avalable on https://drive.google.com/drive/folders/11TmSXZFtwiXhy1kyQdvEvzc5Mu-cHI7S?usp=share_link.

- **Advertisers and ad requests initialization**. The simulator first randomly generate some advertisers, who are categorized into 20 types denotes the varying business conditions of advertisers. Each advertisers possesses a total budget, expected CPC, historical CTR, CVR, GMV etc., which are sampled from a Gaussian distribution based on their respective category. Then, the simulator initializes the total ad requests in a day, each of which belongs to one of 4 channels. Note that the number of ad requests and arrival distribution of each channel are kept relatively consistent with the online platform. Subsequently, the simulator replays these ad requests in chronological order and simulates advertisers' auction as follow.

- **Advertisers retrieval.** Each ad request represents a browsing from a user, thus the simulator randomly generates a category preference of advertisers and selects a few advertisers belonging to that category to participate in the auction for that ad request.

- **Advertisers bidding and ranking.** The base bidding strategy deployed in the simulator is the CTR-based bidding (i.e., bidding with $P(click|ad, advertiser)*CPC_m^{set}$). The simulator applies that bidding strategy to all of the selected advertisers and sorts them in descending order based on their bidding price. The advertiser with the highest bidding wins this ad and have the chance to display ad. Note that the simulator deployed GSP auction with CPC pricing same as the online platform, which means that only if the user clicks ad, the displayed advertiser will be charged with the price of the second-highest bid.

- **User feedback simulation.** After the ad auction is finished, the simulator samples the user feedback towards the displayed ads from multiple Gaussian distribution, including whether the user clicks, makes order, and the making order amount. Then we update the advertiser's daily statistics (i.e., real-time CTR, CVR, budget consumption, the number of clicks, etc), to simulate the real-time feature on the platform.

## :clap: Reference
We express gratitude for the open-source code of the baseline methods mentioned in our paper, which are listed below:
- CBRL (https://github.com/HaozheJasper/CBRL_KDD22)
- MCQ (https://github.com/dmksjfl/MCQ)
- CQL (https://github.com/aviralkumar2907/CQL)

## :e-mail: Contact
If you have any question, please email `wanghao@bit.edu.cn`.

