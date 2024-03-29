# AutoSF
The code for our paper conference paper ["AutoSF: Searching Scoring Functions for Knowledge Graph Embedding"](https://arxiv.org/pdf/1904.11682.pdf) in ICDE 2020 and the journal extension [AutoSF+: "Bilinear Scoring Function Search for Knowledge Graph Learning"](https://arxiv.org/pdf/2107.00184.pdf) in TPAMI 2022.

News: 

(2022.3) AutoSF+ has been accepted as a research paper in TPAMI!

(2021.4) [AutoSF-OGB](https://github.com/AutoML-4Paradigm/AutoSF/tree/AutoSF-OGB) for Open Graph Benchmark is released.

<p align="center">
    <img src="./figs/biokg-leaderboard.png" width="750" />
</p>

<p align="center">
    <img src="./figs/wikikg2-leaderboard.png" width="750" />
</p>

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper

    @article{zhang2022bilinear,
          title={Bilinear Scoring Function Search for Knowledge Graph Learning},
          author={Zhang, Yongqi and Yao, Quanming and Kwok, James T},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          year={2022},
          publisher={IEEE}
    }

## Instructions
For the sake of ease, a quick instruction is given for readers to reproduce the whole process.
Note that the programs are tested on Linux(Ubuntu release 16.04), Python 3.7 from Anaconda 4.5.11.

Install PyTorch (>0.4.0)
    
    conda install pytorch -c pytorch

Get this repo

    git clone https://github.com/yzhangee/AutoSF
    cd AutoSF
    tar -zvxf KG_Data.tar.gz 

Reproducing the searching/fine-tuning/evaluation procedure, please refer to the bash file "run.sh"
    
    bash run.sh


Explaination of the searched SFs in the file "searched_SFs.txt": 

- The first 4 values (a,b,c,d) represent h_1 * r_1 * t_a + h_2 * r_2 * t_b + h_3 * r_3 * t_c + h_4 * r_4 * t_d. 

- For the others, every 4 values represent one adding block: index of r, index of h, index of t, the sign s.

You can also rely on the "evaluate.py" file to evaluate the searched SFs by manually setting the struct variable.



Related AutoML papers (ML Research group in 4Paradigm)
- Interstellar: Searching Recurrent Architecture for Knowledge Graph Embedding. NeurIPS 2020 [paper](https://arxiv.org/pdf/1911.07132.pdf)[code](https://github.com/AutoML-4Paradigm/Interstellar)
- Efficient Neural Interaction Functions Search for Collaborative Filtering. WWW 2020 [paper](https://arxiv.org/pdf/1906.12091.pdf) [code](https://github.com/xiangning-chen/SIF)
- Efficient Neural Architecture Search via Proximal Iterations. AAAI 2020. [paper](https://arxiv.org/abs/1905.13577) [code](https://github.com/xujinfan/NASP-codes)
- Simple and Automated Negative Sampling for Knowledge Graph Embedding. ICDE 2019 [paper](https://arxiv.org/abs/1812.06410) [code](https://github.com/yzhangee/NSCaching)
- Taking Human out of Learning Applications: A Survey on Automated Machine Learning. Arxiv 2018 [paper](https://arxiv.org/abs/1810.13306)
