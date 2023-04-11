# HiBid


This is the code accompanying the paper: "HiBid: Hierarchical Offline Deep Reinforcement Learning for Cross-Channel Constrained Bidding with Budget Allocation"。
## :page_facing_up: Description
Online display advertising platforms service millions of advertisers by providing real-time bidding (RTB) for billions of ad requests every day. The bidding strategy handles ad requests across channels to maximize the number of clicks under the set financial constraints, i.e., total budget and cost-per-click (CPC), etc. Different from existing works mainly focusing on single channel bidding, we explicitly consider cross-channel constrained bidding with budget allocation. Specifically, we propose a hierarchical offline deep reinforcement learning (DRL) framework called ``HiBid'', based on the state-of-the-art offline DRL approach MCQ with three contributions: (a) auxiliary batch loss for non-competitive budget allocation, (b) 
-generalization for adaptive bidding strategy in response to changing budgets, and (c) CPC-guided action selection to satisfy cross-channel CPC constraint. Through extensive experiments on both the large-scale log data and online A/B testing, we confirm that HiBid outperforms five baselines in terms of the number of clicks, CPC satisfactory ratio, and return-on-investment (ROI). We also deploy HiBid on Meituan advertising platform to already service tens of thousands of advertisers every day. 
## :wrench: Dependencies
- Python == 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch == 1.11.0](https://pytorch.org/)
### Installation
1. Clone repo
    ```bash
    git clone https:xxx
    cd xxx
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

## :clap: Reference

## :e-mail: Contact
If you have any question, please email `wanghao@bit.edu.cn`.

