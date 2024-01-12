## SSE- Stress Specific Word Emebdding Pre-trained Language Model
Code and dataset of the paper:"[Contrastive Learning of Stress-specific Word Embedding for Social Media based Stress Detection](https://dl.acm.org/doi/pdf/10.1145/3580305.3599795)"

### Quick Start
First run the pre-training to pre-training the PLM and get the stress-specific embedding(SSE)
```shell
python contrastive_pretrain_cluster.py
```
Then evaluate SSE
```shell
python main.py
```
### Data
To receive access, you will need to read, sign, and send back the attached data and code usage agreement (DCUA).

The DCUA contains restrictions on how you can use the data. We would like to draw your attention to several restrictions in particular:

- No commercial use.
- You cannot transfer or reproduce any part of the data set. This includes publishing parts of users' posts.
- You cannot attempt to identify any user in the data set.
- You cannot contact any user in the data set.

If your institution has issues with language in the DCUA, please have the responsible person at your institution contact us with their concerns and suggested modifications.

Once the Primary Investigator has signed the DCUA, the Primary Investigator should email the signed form to wangxin_6961@163.com

The stressor and stressful emotion lexicons could be directly found in [Stress Lexicons](https://github.com/xin-wang18/stress_lexicons)
